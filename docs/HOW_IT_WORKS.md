# How this project works, in plain words

This file explains the whole project simply, from the X-ray to the running service. Read it before any interview where this project might come up.

---

## 1. What problem does it solve?

A chest X-ray can show several problems at the same time. This project trains a model that looks at one chest X-ray and gives a probability for each of five findings:

| Finding | In simple words |
| --- | --- |
| Atelectasis | Part of the lung has collapsed |
| Cardiomegaly | The heart looks enlarged |
| Consolidation | Part of the lung is filled with fluid or pus, as in pneumonia |
| Edema | Fluid has built up inside the lungs |
| Pleural Effusion | Fluid has collected around the lungs |

Because one image can have several findings, this is **multi-label** classification: five separate yes/no questions, not one pick-a-class question.

It is a research and portfolio project, **not** a medical device.

---

## 2. The data

- **Dataset:** Stanford **CheXpert**, about 224,000 chest X-rays from Stanford Hospital.
- **Training labels** were produced by a program that read the radiologists' written reports. Each finding is labelled **1** (present), **0** (absent), **-1** (uncertain) or **blank** (not mentioned).
- **How the uncertain labels were handled:** blank became 0, and any image with a -1 for one of the five findings was **dropped**. This is the "U-Ignore" policy from the CheXpert paper. That left **159,329 training images**.
- **Validation set:** **234 images** labelled directly by radiologists, so these labels are much cleaner than the training labels.

**Interview line:** "I trained on 159K CheXpert images using the U-Ignore policy for uncertain labels, and evaluated on the 234-image radiologist-labelled validation set."

---

## 3. The model

- **DenseNet121**, a convolutional neural network already trained on ImageNet (everyday photos). This is called **transfer learning**: it already knows edges and textures, so it learns X-rays faster.
- Its last layer (1000 ImageNet classes) is replaced with a layer that has **5 outputs**, one per finding.
- Each output goes through a **sigmoid**, which turns it into an independent probability between 0 and 1.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight`. Rare findings get extra weight so the model doesn't just learn to say "no" all the time.
- **Training setup:** Adam optimiser, learning rate 1e-4, 10 epochs, batch size 32, 224×224 images, about 4.5 hours on a free Colab T4 GPU.
- **Augmentation:** random horizontal flips and small rotations, so the model doesn't memorise exact pixel positions.

---

## 4. The results, and the honest caveat

| Finding | AUROC |
| --- | --- |
| Atelectasis | 0.818 |
| Cardiomegaly | 0.855 |
| Consolidation | 0.897 |
| Edema | 0.906 |
| Pleural Effusion | 0.919 |
| **Mean** | **0.879** |

**What AUROC means:** pick one sick image and one healthy image at random. AUROC is the chance the model gives the sick one a higher score. 0.5 is a coin flip and 1.0 is perfect.

**The caveat (say this yourself before anyone asks):**
- In the first version, the "best" epoch was chosen using the **same** validation set the score was reported on. That makes 0.879 slightly optimistic. Across epochs 6–10 the score moved between **0.854 and 0.879**.
- 234 images is small, so each AUROC has roughly ±0.03–0.06 of noise.
- **The fix, already in the code:** `train.py` now holds out 5% of training **patients** to pick the best epoch. The validation set is used only once, at the end. `evaluate.py` reports **95% confidence intervals** using bootstrapping.

**Interview line:** "My first version had a selection leak. I picked the checkpoint on the validation set. I fixed it with a patient-level tuning split and added bootstrap confidence intervals."

**Why split by patient?** One patient often has several X-rays. If some of them are in training and others in tuning, the model can "recognise the patient" and the score looks better than it really is.

---

## 5. Grad-CAM: what is the model looking at?

Grad-CAM produces a heatmap over the X-ray showing which regions pushed the model toward a finding. For example, for Pleural Effusion it should highlight the bottom of the lungs.

How it works, in short:
1. Take the last feature map of the network (7×7 grid × 1024 channels).
2. Measure how much each channel affects the chosen finding's score (the gradient).
3. Weight the channels by that importance, add them up, and scale the result up to image size.

**Caveat:** a heatmap shows where the evidence came from, not that the model is right. It can sometimes highlight tubes, wires or text markers instead of disease.

---

## 6. The code layout

```
src/chexpert_cls/
  config.py      the five findings, image size, model version
  data.py        reads the CSV, handles uncertain labels, splits by patient
  validation.py  checks the CSV before training; checks uploaded images before predicting
  model.py       builds DenseNet121, loads weights (downloads from Hugging Face if needed)
  train.py       full training run: tuning split, mixed precision, optional MLflow logging
  evaluate.py    AUROC with confidence intervals, calibration, sensitivity/specificity
  metrics.py     the maths behind evaluate.py
  gradcam.py     heatmap generation
  predict.py     run the model on any image from the command line
  api.py         the web service (FastAPI)
app.py           simple Gradio demo (runs locally)
scripts/         export_onnx.py (browser model), fetch_samples.py (demo images)
docs/index.html  the website
tests/           automatic tests that run on every push
```

---

## 7. The MLOps part: from model file to running service

**MLOps** is everything that makes a model reliable outside a notebook: repeatable training, versioned models, automatic testing, packaging and monitoring.

### 7.1 Data validation
Before training starts, `validate_csv` checks that the CSV has the right columns, that labels are only 1/0/-1/blank, that no image path appears twice, and that paths look like `patientX/studyY/`. If anything is wrong, training stops immediately instead of failing three hours later.

### 7.2 Experiment tracking (MLflow)
`python -m chexpert_cls.train ... --mlflow` records every training run: all the settings used, the loss and AUROC for every epoch, the final scores, and the model file. Later you can compare runs side by side in the MLflow web UI instead of guessing which notebook produced which number.

### 7.3 Model versioning
- The weights live on the **Hugging Face Hub**, which keeps every version.
- Every API response includes `MODEL_VERSION`, so you always know which model made a prediction.
- `/version` shows the **SHA-256 fingerprint** of the loaded weights file. If `CHEXPERT_WEIGHTS_SHA256` is set, the service **refuses to start** when the file doesn't match, so the wrong model can't be served by accident.

### 7.4 The API (FastAPI)

| Endpoint | What it does |
| --- | --- |
| `GET /health` | "Am I alive?" Docker uses it to restart a broken container |
| `GET /version` | Which code version and which model are running |
| `POST /predict` | Upload an X-ray and get five probabilities, the top finding, warnings and latency |
| `POST /explain` | Upload an X-ray and get the Grad-CAM heatmap as a PNG |
| `GET /drift` | Is the incoming data shifting? (see 7.9) |
| `GET /metrics` | Numbers for monitoring (see 7.7) |

**Input safety:** real chest X-rays are grayscale and roughly square. If someone uploads a colour photo, a tiny thumbnail or a very wide screenshot, the API adds a **warning**. With `CHEXPERT_STRICT_INPUT=1` it **rejects** the image instead. It only accepts PNG and JPEG (error 415 otherwise), stops reading after 10 MB (error 413), and refuses "decompression bombs": small files that unpack into gigantic images.

### 7.5 Docker
The `Dockerfile` packages Python, CPU-only PyTorch, the code and (optionally) the model weights into one **image** that runs the same way on any machine.
- CPU-only PyTorch keeps the image much smaller than the GPU version.
- It runs as a **non-root user**, which is a basic security practice.
- A **healthcheck** calls `/health` every 30 seconds.

`docker compose up` starts the API and an MLflow server together.

### 7.6 CI/CD (GitHub Actions)
- **CI** runs on every push:
  1. **Lint** (ruff) catches unused or undefined names.
  2. **Tests** (pytest) check label handling, the patient split, the metrics, data validation, the model, Grad-CAM and every API endpoint.
  3. **Docker build**, then the container is started and real requests are sent to it (a smoke test).
- **CD:** pushing a git tag like `v1.2.0` builds the image with the weights inside and publishes it to **GitHub Container Registry**, ready to deploy anywhere.

### 7.7 Monitoring
`/metrics` exposes numbers in the Prometheus format, a standard monitoring tool:
- how many requests came in, and how many failed;
- how long predictions take;
- **the distribution of predicted probabilities for each finding**. If that distribution suddenly shifts (say, the model starts calling everything "Edema"), it's a sign the incoming data has changed. This is called **drift**, and it means the model may need retraining.

---

### 7.8 Quality gate: never ship a worse model
`models/baseline_metrics.json` holds the scores of the model currently in use. When you train a new model, `evaluate.py` writes its scores to `valid_report.json`, and the gate compares the two:
- if the **mean AUROC** drops by more than **0.01**, it fails;
- if **any single finding** drops by more than **0.03**, it fails, even if the mean went up.

Why both rules? A new model can look better on average while getting much worse at one disease. Missing one disease is exactly the kind of mistake that matters in medicine.

In CI, commit the new report as `models/candidate_metrics.json` and the pipeline runs the gate automatically. A failed gate means a red pipeline, so the model doesn't get promoted.

### 7.9 Drift detection (PSI)
During validation we record how the model's probabilities are spread out for each finding: for example, "most Edema scores are below 0.2". That's the **reference**. In production, the API keeps the last 500 predictions and compares their spread with the reference using the **Population Stability Index (PSI)**:
- PSI below 0.1 means **stable**;
- 0.1 to 0.25 means a **moderate shift**, so keep an eye on it;
- above 0.25 means a **major shift**: the incoming images are probably different (a new hospital, new scanner, or the wrong kind of image).

You can see it at `/drift`, and it feeds into Prometheus as `chexpert_drift_psi`.

Why watch predictions instead of accuracy? In production nobody tells you the right answer straight away, so accuracy can't be measured live. A change in what the model predicts is the earliest warning you get.

### 7.10 Prometheus, alerts and Grafana
`docker compose up` starts the full monitoring stack:
- **Prometheus** collects the `/metrics` numbers every 15 seconds and checks four **alert rules**: the API is down, more than 10% of requests fail, predictions slower than 2 seconds (95th percentile), and drift above 0.25.
- The alert rules have their own **unit tests** (`alerts_test.yml`). CI feeds them fake data, such as "drift stays at 0.30 for 20 minutes", and checks that the right alert fires. It also checks that alerts don't fire too early.
- **Grafana** shows a ready-made dashboard: traffic, error ratio, response time, average prediction per finding, and drift with warning lines at 0.1 and 0.25.
- A Python test checks that every metric name used in the dashboard and alerts really exists in the API, so a renamed metric can't silently break monitoring.

### 7.11 Checking a new dataset before trusting the model
The drift monitor in 7.9 watches live traffic. But often the real question comes earlier: *"a new hospital sent us 5,000 X-rays; will the model work on them?"* That's what `drift_check` answers:

```bash
python -m chexpert_cls.drift_check new_images/ --out drift_report/
```

It runs the model on every image in the folder and reports three things:
1. **Input quality:** how many images look wrong (colour, tiny, odd shape).
2. **Prediction drift:** PSI per finding, compared with the validation reference. A big shift means "these images are different from what the model was tested on".
3. **External validation (if labels exist):** AUROC per finding with confidence intervals. This is the honest answer to "how good is the model on this data?". It reads a simple labels CSV, or the NIH ChestX-ray14 label file directly.

Notebook 02 runs this on **NIH ChestX-ray14**, a different hospital with a different labeller. Expect the AUROC to drop compared with CheXpert. That drop is normal, and reporting it is what separates a careful ML engineer from someone quoting one number.

### 7.12 The website: the model in your browser
Live at https://atomichalifax.github.io/ChestXRay-Disease-Classification/

- `pages.yml` converts the PyTorch model to **ONNX**, a format that runs in a web browser.
- Before publishing, it **checks the ONNX model gives the same answers as PyTorch** (within 0.0001) and the same Grad-CAM heatmap. If not, the release stops.
- Trick worth explaining in an interview: for DenseNet, Grad-CAM can be written as plain maths on the last layer (`ReLU(weights × feature maps)`), so it's baked into the ONNX file and the browser needs no gradients.
- The page reads GitHub's public API to show the **live pipeline**: whether the latest tests and release passed.
- Cost: $0. GitHub Pages hosts it, and the visitor's device does the computing.

### 7.13 Security
- **No secrets** anywhere in the repo or website. `.env` is ignored by git.
- **Uploads:** PNG/JPEG only, 10 MB cap, decompression-bomb guard.
- **Rate limit:** max 30 uploads per minute per IP, then error 429.
- **CORS:** only our website and localhost may call the API from a browser.
- **Errors** never show stack traces.
- **Dependabot** opens weekly PRs to update packages, and CI must pass before merging.
- There's no database, login or file storage, so SQL injection, auth and bucket risks don't apply.

Full checklist: [`MLOPS.md`](../MLOPS.md).

## 8. Limitations (know these well)

1. **One hospital only.** It might not work as well on X-rays from other hospitals, scanners or countries. The next step is testing on another dataset (MIMIC-CXR or NIH).
2. **Noisy training labels.** They came from a text-reading program, not from radiologists.
3. **Small validation set** (234 images), so the scores have wide error bars.
4. **Threshold 0.5 is arbitrary.** A real product would pick a separate cut-off per finding, depending on whether missing a disease or raising a false alarm is worse.
5. **Grad-CAM is not proof.** It shows where the model looked, not that it's correct.

---

**What would you do next?**
Retrain with the fixed protocol, compare U-Ones vs U-Ignore, use 320-pixel images, test on an external dataset, and calibrate the probabilities.

**How would you know if the deployed model gets worse?**
Watch the prediction distribution in `/metrics` for drift, sample predictions for radiologist review, and retrain and redeploy through the same CI/CD pipeline with a new `MODEL_VERSION`.
