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
app.py           simple Gradio web demo
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
| `GET /metrics` | Numbers for monitoring (see 7.7) |

**Input safety:** real chest X-rays are grayscale and roughly square. If someone uploads a colour photo, a tiny thumbnail or a very wide screenshot, the API adds a **warning**. With `CHEXPERT_STRICT_INPUT=1` it **rejects** the image instead. It also rejects files that aren't images (error 415) and files over 10 MB (error 413).

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

## 8. Limitations (know these well)

1. **One hospital only.** It might not work as well on X-rays from other hospitals, scanners or countries. The next step is testing on another dataset (MIMIC-CXR or NIH).
2. **Noisy training labels.** They came from a text-reading program, not from radiologists.
3. **Small validation set** (234 images), so the scores have wide error bars.
4. **Threshold 0.5 is arbitrary.** A real product would pick a separate cut-off per finding, depending on whether missing a disease or raising a false alarm is worse.
5. **Grad-CAM is not proof.** It shows where the model looked, not that it's correct.

---

## 9. Likely interview questions, with short answers

**Why DenseNet121?**
It's the backbone used in the CheXpert and CheXNet papers. It's well-studied on chest X-rays, small enough for a free GPU, and good with transfer learning.

**Why AUROC and not accuracy?**
Most images are negative for any one finding, so a model that always says "no" gets high accuracy. AUROC measures how well the model ranks sick above healthy, regardless of threshold.

**Why sigmoid and not softmax?**
The findings aren't mutually exclusive. Softmax forces the probabilities to sum to 1, while sigmoid treats each finding as its own yes/no question.

**What would you do next?**
Retrain with the fixed protocol, compare U-Ones vs U-Ignore, use 320-pixel images, test on an external dataset, and calibrate the probabilities.

**How would you know if the deployed model gets worse?**
Watch the prediction distribution in `/metrics` for drift, sample predictions for radiologist review, and retrain and redeploy through the same CI/CD pipeline with a new `MODEL_VERSION`.
