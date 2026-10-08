# MLOps: what happens to this model after training

**Live site:** https://atomichalifax.github.io/ChestXRay-Disease-Classification/ (demo, live pipeline status, model card)

Training a DenseNet121 took one Colab run. Everything below is what makes that model safe to change, ship and watch. Each part is in the repo and runs in CI unless marked **not run yet**.

```
 train.csv ─► validate_csv ─► train.py (--mlflow) ─► best.pth ─► Hugging Face Hub (SHA-256 pinned)
                                                        │
 evaluate.py ─► valid_report.json ─► QUALITY GATE ◄─ models/baseline_metrics.json
             └► valid_predictions.npz ─► drift reference
                                                        │
 git push ─► CI ─┬─ ruff + pytest (+ gate if a candidate report is committed)
                 ├─ promtool: config check + alert-rule unit tests
                 └─ docker build ─► container smoke test (/health /version /drift /predict /explain)
          └► Website ─► ONNX export ─► parity check vs PyTorch + Grad-CAM ─► GitHub Pages
 git tag v* ─► Release ─► image with weights baked in ─► ghcr.io
                                                        │
 FastAPI ─► /metrics ─► Prometheus (4 alert rules) ─► Grafana
        └► /drift (PSI of recent predictions vs validation)
```

---

## 1. Data validation
`validation.validate_csv` checks columns, label values (1/0/-1/blank), duplicate paths and the `patientX/studyY/` layout before training starts. A bad CSV fails in seconds instead of three hours into a GPU run.

## 2. Experiment tracking
`python -m chexpert_cls.train ... --mlflow` logs parameters, per-epoch loss and AUROC, final metrics and the checkpoint. `docker compose up` starts an MLflow server on :5000.

## 3. Honest model selection
v1.0 picked its best epoch on the validation set, so its 0.879 mean AUROC is optimistic (0.854–0.879 across epochs 6–10). v1.1 code selects on a **5% patient-disjoint split of train**, and `evaluate.py` reports bootstrap 95% CIs, AP, ECE and sensitivity/specificity. **v1.1 retrain: not run yet** (notebook 02).

## 4. Model versioning
- Weights live on the Hugging Face Hub: [`AtomicHalifax/ChestXRay-DenseNet121`](https://huggingface.co/AtomicHalifax/ChestXRay-DenseNet121).
- `MODEL_VERSION` (`config.py`) is returned on every API response.
- `/version` shows the SHA-256 of the loaded weights. With `CHEXPERT_WEIGHTS_SHA256` set, the service **refuses to start** on a mismatch.

## 5. Quality gate
`python -m chexpert_cls.gate --candidate <report>` compares a new model against `models/baseline_metrics.json` and fails if:
- mean AUROC drops by more than **0.01**, or
- **any** finding drops by more than **0.03**, even if the mean improved.

Commit a report as `models/candidate_metrics.json` and CI runs the gate; a red pipeline means the model isn't promoted.

## 6. CI/CD (GitHub Actions)

| Workflow | Trigger | What it proves |
| --- | --- | --- |
| `ci.yml` → test | every push / PR | lint, unit + API tests, quality gate |
| `ci.yml` → monitoring | every push / PR | Prometheus config valid, alert rules fire (and don't fire early) on fake data |
| `ci.yml` → docker | after tests | image builds, container starts, real requests succeed |
| `pages.yml` | changes to site / model code | ONNX export matches PyTorch (probs within 1e-4) and Grad-CAM (max diff < 1e-3), then deploys |
| `release.yml` | tag `v*` | publishes `ghcr.io/atomichalifax/chexpert-api` with weights inside |
| Dependabot | weekly | PRs for outdated pip packages, Actions and the Docker base image; CI must pass first |

## 7. Serving
- **API** (`api.py`, FastAPI): `/health`, `/version`, `/predict`, `/explain` (Grad-CAM PNG), `/drift`, `/metrics`.
- **Docker:** CPU-only torch, non-root user, healthcheck, optional baked-in weights.
- **Browser build:** the website runs the same model as ONNX in the visitor's browser (ONNX Runtime Web). Grad-CAM is baked into the graph: for DenseNet's ReLU → avg-pool → linear head, Grad-CAM equals `ReLU(Σ W_ck · A_k)`, so no gradients are needed. No server, no cost, and the image never leaves the device.

## 8. Monitoring
- **Prometheus metrics:** request counts by status, latency histogram, predicted-probability histogram per finding, drift gauge.
- **Drift (PSI):** the API keeps the last 500 predictions and compares each finding's distribution with the validation reference. <0.1 stable, 0.1–0.25 moderate, >0.25 major. Ground truth isn't available live, so prediction drift is the earliest warning.
- **Alerts** (unit-tested with `promtool`): API down, error rate >10%, p95 latency >2 s, PSI >0.25.
- **Grafana dashboard:** traffic, error ratio, p50/p95 latency, mean prediction per finding, drift with 0.1/0.25 lines.
- `tests/test_monitoring.py` fails if a dashboard or alert uses a metric the API doesn't export.

## 9. Checking a new dataset first
```bash
python -m chexpert_cls.drift_check new_images/ --labels labels.csv --fail-on-shift
```
Reports input-quality problems, PSI per finding, and (with labels, including NIH ChestX-ray14 format) AUROC with 95% CIs. **NIH external validation: not run yet** (notebook 02).

## 10. Security

| Area | What's in place |
| --- | --- |
| Secrets | No keys in the repo or the website. `.env` is git-ignored and was never committed (history checked). The site only calls GitHub's public, read-only API. |
| Uploads | PNG/JPEG only; reads at most 10 MB (`CHEXPERT_MAX_UPLOAD_MB`); rejects decompression bombs (>100 MP) |
| Input checks | Colour, tiny or oddly shaped images are flagged (or rejected with `CHEXPERT_STRICT_INPUT=1`); unknown findings → 422 |
| Abuse | 30 POSTs/min per IP (`CHEXPERT_RATE_LIMIT`) → 429 |
| CORS | Only the project site and localhost (`CHEXPERT_CORS_ORIGINS`) |
| Errors | No stack traces in responses |
| Container | non-root user, CPU-only, healthcheck |
| Dependencies | Dependabot weekly PRs, gated by CI |
| Website | Third-party text (image credits) is escaped before rendering |

No database, no logins, no user storage, so there's no SQL, auth or bucket to secure.

## 11. Cost
$0. Website and model on GitHub Pages, CI on GitHub Actions, weights on Hugging Face, monitoring stack runs locally with `docker compose up`.

## 12. What's next
1. Run notebook 02: v1.1 retrain → gate → new `MODEL_VERSION` → drift reference → NIH external validation.
2. Per-finding thresholds and calibration (temperature scaling).
3. int8 quantisation of the browser model, with the same parity check in CI.

See [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) for a plain-language walkthrough.
