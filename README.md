# Chest X-ray Finding Classification (CheXpert, DenseNet121)

**[Live demo →](https://atomichalifax.github.io/ChestXRay-Disease-Classification/)** · [MLOps write-up](MLOPS.md) · [Model weights](https://huggingface.co/AtomicHalifax/ChestXRay-DenseNet121)

[![CI](https://github.com/AtomicHalifax/ChestXRay-Disease-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/AtomicHalifax/ChestXRay-Disease-Classification/actions/workflows/ci.yml)
[![Website](https://github.com/AtomicHalifax/ChestXRay-Disease-Classification/actions/workflows/pages.yml/badge.svg)](https://atomichalifax.github.io/ChestXRay-Disease-Classification/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License](https://img.shields.io/badge/license-MIT-green)

A multi-label classifier for five chest X-ray findings: **Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion**. It uses a DenseNet121 fine-tuned on 159,329 Stanford CheXpert images, with Grad-CAM explanations, wrapped in a full MLOps pipeline: data validation, experiment tracking, a quality gate that blocks worse models, drift monitoring, a hardened API, CI/CD, and a browser build checked against PyTorch before every release.

The demo runs the model in your browser (ONNX), so images never leave your device. The page also shows the live CI/CD status and the model card.

<img width="1672" height="941" alt="Project overview" src="https://github.com/user-attachments/assets/1c646a7c-247d-4e5e-910a-8df9b88fa4a8" />

> Research and portfolio project. Not a medical device and not for clinical use.

---

## Results (v1.0 checkpoint)

These are measured on the official CheXpert validation set: **234 images**, labelled by radiologist consensus.

| Finding          | AUROC  | Precision @0.5 | Recall @0.5 | F1 @0.5 |
| ---------------- | -----: | -------------: | ----------: | ------: |
| Atelectasis      | 0.818  | 0.62 | 0.73 | 0.67 |
| Cardiomegaly     | 0.855  | 0.74 | 0.59 | 0.66 |
| Consolidation    | 0.897  | 0.43 | 0.79 | 0.56 |
| Edema            | 0.906  | 0.51 | 0.87 | 0.64 |
| Pleural Effusion | 0.919  | 0.80 | 0.73 | 0.77 |
| **Mean**         | **0.879** | | | |

**Read this before quoting the number.** In v1.0 the best epoch was chosen using this same validation set. Across epochs 6–10 the mean AUROC moved between **0.854 and 0.879**, so 0.879 is the most favourable of five looks at a small set. With only 234 images, per-finding AUROCs also carry roughly ±0.03–0.06 of sampling noise. The v1.1 code below fixes the selection leak (see [Protocol](#protocol)). `evaluate.py` reports bootstrap 95% CIs for any checkpoint.

<p align="center">
  <img src="images/ROC%20Curves%20for%20Densenet%20121%20CheXpert.png" width="48%" alt="ROC curves">
  <img src="images/Grad-CAM%20Visualization.png" width="48%" alt="Grad-CAM">
</p>

More figures (PR curves, confusion matrices, failure cases) are in [`images/`](images/), and per-finding notes are in [`docs/`](docs/).

---

## Quick start

```bash
git clone https://github.com/AtomicHalifax/ChestXRay-Disease-Classification
cd ChestXRay-Disease-Classification
pip install -e ".[demo]"

# Predict on your own image (weights auto-download from Hugging Face)
python -m chexpert_cls.predict xray.jpg --gradcam out/

# Local Gradio demo with Grad-CAM
python app.py
```

Weights: [`AtomicHalifax/ChestXRay-DenseNet121`](https://huggingface.co/AtomicHalifax/ChestXRay-DenseNet121) on Hugging Face.

### Serve it (API + Docker)

```bash
docker compose up --build          # API :8000 (/docs), MLflow :5000, Prometheus :9090, Grafana :3000
curl -F file=@xray.jpg localhost:8000/predict
curl -F file=@xray.jpg -o cam.png "localhost:8000/explain?finding=Edema"
```

| Endpoint | What it does |
| --- | --- |
| `GET /health` | Liveness check, used by the Docker healthcheck |
| `GET /version` | API version, model version, SHA-256 of the loaded weights |
| `POST /predict` | Five probabilities, the top finding, input-quality warnings, latency |
| `POST /explain` | Grad-CAM overlay PNG for one finding |
| `GET /drift` | Drift status: PSI per finding over recent predictions (`stable` / `moderate_shift` / `major_shift`) |
| `GET /metrics` | Prometheus metrics: request counts, latency, prediction distribution (for drift monitoring) |

## MLOps

Full write-up: **[MLOPS.md](MLOPS.md)**.

```
 train.csv ──► validate_csv ──► train.py ──► MLflow (params, metrics, best.pth)
                                   │
                                   ▼
                 Hugging Face Hub (versioned weights, SHA-256 pinned)
                                   │
 evaluate.py ──► valid_report.json ──► quality gate (vs models/baseline_metrics.json)
             └─► valid_predictions.npz ──► drift reference (models/drift_reference.json)
                                   │
 git push ──► CI: ruff + pytest + gate ──► docker build ──► container smoke test
          └─► promtool: check config + unit-test alert rules
          └─► Website: ONNX export ──► parity check ──► GitHub Pages
 git tag v* ──► Release: build image (weights baked in) ──► ghcr.io/<owner>/chexpert-api
                                   │
                                   ▼
   FastAPI service ──► /metrics ──► Prometheus (alert rules) ──► Grafana dashboard
                  └─► /drift (PSI of recent predictions vs validation reference)
```

- **Data validation:** `validate_csv` checks columns, label values, duplicate paths and path layout before training starts.
- **Experiment tracking:** `train.py --mlflow` logs every run's params, per-epoch metrics, final validation AUROCs and artifacts.
- **Model versioning:** `MODEL_VERSION` is on every API response. Setting `CHEXPERT_WEIGHTS_SHA256` stops the service from starting if the weights file differs.
- **Input guard:** the API warns about (or, with `CHEXPERT_STRICT_INPUT=1`, rejects) images that are colour, tiny or oddly shaped.
- **Quality gate:** `python -m chexpert_cls.gate --candidate <report>` fails if mean AUROC drops more than 0.01, or any finding drops more than 0.03, against the committed baseline. CI runs it automatically when `models/candidate_metrics.json` is committed.
- **Drift detection:** the API keeps the last 500 predictions and compares each finding's probability distribution with the validation set, using the Population Stability Index (PSI). It's exposed at `/drift` and as the `chexpert_drift_psi` metric. Build the reference with `python -m chexpert_cls.drift --predictions results/valid_predictions.npz`.
- **Monitoring:** `docker compose up` also starts Prometheus (:9090) and Grafana (:3000), with a pre-built dashboard covering traffic, error ratio, p50/p95 latency, mean prediction per finding and drift. Four alert rules (API down, error rate >10%, p95 latency >2 s, PSI >0.25) are checked and unit-tested with `promtool` in CI.
- **CI/CD:** every push runs lint, unit and API tests, builds the image and smoke-tests the container. A `v*` tag publishes the image to GHCR. Dependabot opens weekly update PRs, gated by CI.
- **Browser release:** `pages.yml` exports the model to ONNX, checks it against PyTorch and Grad-CAM, and deploys the [website](https://atomichalifax.github.io/ChestXRay-Disease-Classification/).
- **Security:** PNG/JPEG only, 10 MB read cap, decompression-bomb guard, 30 POSTs/min per IP, CORS locked to the site + localhost, no secrets in the repo. Details in [MLOPS.md](MLOPS.md#10-security).

- **Batch drift check + external validation:** before trusting the model on a new dataset, run `python -m chexpert_cls.drift_check new_images/`. It reports input-quality problems, PSI drift per finding, and, given labels (a generic CSV or NIH ChestX-ray14 format), AUROC with 95% CIs on that dataset. `--fail-on-shift` makes it exit non-zero on a major shift.

A plain-language walkthrough of the whole project is in [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

### Try it with sample images

```bash
make samples          # 20 openly licensed chest X-rays (Wikimedia Commons) + 4 deliberately bad inputs
python -m chexpert_cls.drift_check samples/xray/
for f in samples/xray/*.png samples/bad_inputs/*; do curl -s -F file=@$f localhost:8000/predict; echo; done
```

Credits and licences are in `samples/manifest.json`. The "expected" findings come from the uploaders' descriptions; they are not verified labels. To compare against real radiologist labels, `scripts/make_test_set.py` copies a small labelled set out of *your own* CheXpert download (research licence: keep it private).

### Reproduce training and evaluation

Get **CheXpert-v1.0-small** from [Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) (or the [Kaggle mirror](https://www.kaggle.com/datasets/ashery/chexpert)). Point `--data-root` at the folder containing `train.csv` and `valid.csv`.

```bash
python -m chexpert_cls.train    --data-root data/CheXpert-v1.0-small --epochs 10 --policy ignore --pos-weight
python -m chexpert_cls.evaluate --data-root data/CheXpert-v1.0-small --weights runs/densenet121/best.pth
```

Or run everything on Colab with [`notebooks/02_retrain_and_external_validation.ipynb`](notebooks/02_retrain_and_external_validation.ipynb): retrain with MLflow tracking, evaluate, quality gate, build the drift reference, and test on NIH ChestX-ray14.

---

## Protocol

| | v1.0 (notebook) | v1.1 (`src/`) |
| --- | --- | --- |
| Training images | 159,329 of 223,414. Rows with any uncertain (-1) target dropped (U-Ignore). | Same by default; `--policy ones/zeros` available |
| Views | Frontal + lateral | Same; `--frontal-only` available |
| Loss | BCE with per-class `pos_weight` | Same (`--pos-weight`) |
| Optimiser | Adam 1e-4, 10 epochs, batch 32, 224 px | Adam 1e-4 + cosine decay, AMP on GPU, fixed seed |
| Checkpoint selection | Mean AUROC on the **validation set** (leaks) | Mean AUROC on a **5% patient-disjoint split of train** |
| Reporting | Point AUROC | AUROC + bootstrap 95% CI, AP, ECE, sens/spec |
| Hardware | 1× Tesla T4 (Colab), ~4.5 h | Same |

This is a reproduction of the DenseNet121 U-Ignore baseline from the CheXpert paper (Irvin et al., 2019), at lower resolution (224 vs 320 px) and with a single model instead of an ensemble. Use the paper's Table 2 for a like-for-like reference.

## Repository layout

```
src/chexpert_cls/   config, data (label policies, patient split), validation, model, train, evaluate,
                    predict, gradcam, metrics, gate (quality gate), drift (PSI), drift_check (batch drift +
                    external validation), api (FastAPI service)
models/             baseline_metrics.json (gate baseline); drift_reference.json once built
app.py              Gradio demo (local)
tests/              pytest: labels, patient split, metrics, validation, model, Grad-CAM, API contract
Dockerfile          CPU inference image (non-root, healthcheck, optional baked weights)
docker-compose.yml  API + MLflow + Prometheus + Grafana
monitoring/         Prometheus config, alert rules + their unit tests, Grafana dashboard
scripts/            fetch_samples, make_test_set, export_onnx (browser model + parity check)
samples/            manifest.json of demo images (images themselves are downloaded, not committed)
docs/index.html     the website (GitHub Pages)
.github/workflows/  ci.yml (lint, tests, docker smoke test), pages.yml (ONNX + website), release.yml (GHCR image)
.github/dependabot.yml  weekly dependency update PRs
notebooks/          original exploration (00) and evaluation (01); 02 = Colab retrain + external validation
docs/               website, plus dataset, architecture, training, evaluation and limitations notes
images/             evaluation figures
```

## Limitations

- Trained and evaluated only on CheXpert: a single hospital, mostly adult inpatients, many AP portable films. External validation on NIH ChestX-ray14 is set up (notebook 02) but has not been run yet.
- The training labels come from an NLP labeller run on radiology reports, so they're noisy. The validation set is small (234 images).
- Grad-CAM shows where the model's evidence is. It doesn't show that the model is right, and it can highlight support devices or text markers.
- A threshold of 0.5 is arbitrary. Operating points should be tuned per finding on held-out data.

More detail is in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Next steps

- [x] Live website with in-browser inference and live pipeline status
- [ ] Retrain with v1.1 protocol and report CIs
- [ ] U-Ones vs U-Ignore ablation, 320 px input
- [ ] Run notebook 02: v1.1 retrain + NIH ChestX-ray14 external validation
- [ ] Per-finding calibration (temperature scaling) and operating points
- [ ] int8 browser model with the same parity check

## References

- Irvin et al. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI 2019.
- Huang et al. *Densely Connected Convolutional Networks.* CVPR 2017.
- Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.

Built by **Athrva Raval** (BSc Data Science & AI, IIT Guwahati). [Website](https://atomichalifax.github.io/ChestXRay-Disease-Classification/) · [LinkedIn](https://www.linkedin.com/in/athrva-raval/)

MIT License. CheXpert data is subject to Stanford's own research-use agreement.
