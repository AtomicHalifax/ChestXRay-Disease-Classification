# CLAUDE.md

Context for AI coding assistants working on this repo. Keep it current when the project changes.

## What this is

Multi-label chest X-ray classifier (DenseNet121, Stanford CheXpert) for 5 findings: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion. It's a portfolio project by Athrva Raval (BSc Data Science & AI, IIT Guwahati), used to land a paid remote ML internship. It wraps an ML model in a full MLOps stack. Research use only, not a medical device.

## Facts that must stay accurate (don't inflate them)

- Trained on **159,329** images (U-Ignore: rows with any uncertain -1 target dropped), not 224K. CheXpert has 223,414 training rows in total.
- Validation set: **234 images**. v1.0 mean AUROC **0.879** (per finding 0.818 / 0.855 / 0.897 / 0.906 / 0.919).
- v1.0 picked its best epoch **on the validation set**, so 0.879 is optimistic. Across epochs 6–10 it ranged 0.854–0.879. The README states this openly; keep it that way.
- 224×224 input, Adam 1e-4, 10 epochs, batch 32, pos_weight BCE, one Colab T4 run (~4.5 h).
- Weights: Hugging Face Hub `AtomicHalifax/ChestXRay-DenseNet121`, file `best_densenet121.pth` (state_dict).

## Layout

```
src/chexpert_cls/  config (findings, MODEL_VERSION), data (label policies, patient split), validation (CSV + image checks),
                   model (build/load, HF download), train (patient-level tune split, AMP, --mlflow), evaluate (bootstrap CIs,
                   writes valid_report.json + valid_predictions.npz), metrics, gradcam (autograd.grad, thread-safe),
                   predict (CLI), gate (quality gate), drift (PSI monitor), drift_check (batch drift +
                   external validation, NIH label format), api (FastAPI)
app.py             Gradio demo (local only)
tests/             pytest; torch/fastapi tests use importorskip and CHEXPERT_RANDOM_WEIGHTS=1
models/            baseline_metrics.json (gate baseline); drift_reference.json (not built yet; needs real eval)
monitoring/        prometheus.yml, alerts.yml, alerts_test.yml (promtool unit tests), Grafana provisioning + dashboard
scripts/           fetch_samples.py (20 Wikimedia CXRs + bad inputs; --site writes docs/samples), export_onnx.py (browser model
                   with baked Grad-CAM, asserts parity with PyTorch), make_test_set.py (private CheXpert subset)
samples/           manifest.json only; downloaded images are git-ignored
Dockerfile         CPU torch, non-root, healthcheck, ARG BAKE_WEIGHTS=1 bakes weights in
docker-compose.yml api :8000, mlflow :5000, prometheus :9090, grafana :3000
.github/workflows  ci.yml (test: ruff+pytest+gate | monitoring: promtool | docker: build + container smoke test)
                   pages.yml (ONNX export + parity check + samples → GitHub Pages), release.yml (tag v* → ghcr.io/<owner>/chexpert-api)
.github/dependabot.yml  weekly pip / actions / docker update PRs
docs/index.html    the website (vanilla HTML/JS, ONNX Runtime Web, live pipeline from the public GitHub API)
MLOPS.md           recruiter-facing MLOps write-up incl. security checklist
docs/HOW_IT_WORKS.md  plain-language walkthrough for the owner's interview prep
notebooks/         original v1.0 training (00) and evaluation (01), historical; 02 = Colab retrain + NIH external validation
```

## Commands

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,serve]"
ruff check src tests app.py        # correctness rules only (E9, F)
pytest -q
uvicorn chexpert_cls.api:app --port 8000
docker compose up --build
python -m chexpert_cls.train --data-root <CheXpert-v1.0-small> --pos-weight --mlflow
python -m chexpert_cls.evaluate --data-root <...> --weights <ckpt> --out results/
python -m chexpert_cls.gate --candidate results/valid_report.json
python -m chexpert_cls.drift --predictions results/valid_predictions.npz
```

## Conventions

- Bump `MODEL_VERSION` (config.py) whenever the weights change. Bump the package version in both `pyproject.toml` and `__init__.py`.
- Every metric used in `monitoring/` must exist in `api.py`. `tests/test_monitoring.py` enforces this.
- Alert rule changes need a matching case in `monitoring/prometheus/alerts_test.yml`.
- Keep ruff limited to correctness rules. Newer ruff defaults (RUF100, I001) broke CI runs 1–4.
- CI must stay green. A temporary red between multi-commit pushes is acceptable only when the final commit is green.
- Never use the validation set for model selection or threshold tuning. Use `patient_split` from train.csv.

## Working environment notes (for Claude sessions)

- The cloud sandbox has no PyTorch and no PyPI/HF access, so torch tests only run in GitHub Actions. Pure numpy/sklearn tests can run locally with a small pytest shim.
- There's no git push credential. Changes reach GitHub through the owner's signed-in browser: the `/upload/main/<dir>` page plus a synthetic DragEvent drop, or `/edit/` and `/new/` pages. Dotfiles can't go through upload; use `/new/main?filename=...&value=...`. Always verify afterwards with `git fetch` + `git diff HEAD origin/main` (must be empty), then check the Actions run.
- Patch existing files by fetching `raw.githubusercontent.com` in-page and doing string replacements. That's cheaper than retyping.

## Decisions

- Hugging Face Hub is used **only for weight storage**. No HF Spaces.
- Demo images: Wikimedia Commons (CC0/PD/CC BY-SA, credited, fetched on demand). Never publish CheXpert images (research-only licence). NIH ChestX-ray14 is the external test set.
- The owner doesn't expect real live traffic soon. Batch drift check + external validation matter more for the portfolio than live monitoring.
- No paid domain, everything free. Website: https://atomichalifax.github.io/ChestXRay-Disease-Classification/ (owner chose to keep this URL). Inference runs in the browser via ONNX, so there is no hosted API. AWS was dropped.
- Website: white, MLOps-first. No "only five diseases" wording, no "runs 100% in your browser" tagline, no visitor counter, no run-history bars.
- API security: PNG/JPEG only, capped read, pixel limit, per-IP rate limit, CORS locked (env-configurable). Keep these; tests cover them.
- The owner wants the MLOps built for him, and needs to understand it for interviews. Keep `docs/HOW_IT_WORKS.md` updated in plain language.

## Status (v1.4 + website)

Done and green in CI: package, tests, API (`/health /version /predict /explain /drift /metrics`), input guard, CSV validation, MLflow hooks, SHA-256 weight pinning, Docker + smoke test, quality gate, PSI drift monitor, batch drift check / external validation, Prometheus alert rules with unit tests (visible in the Prometheus UI; no notifications — Alertmanager/email was removed because there's no real traffic), Grafana dashboard, release workflow (v1.4.0 published, GHCR image built).

Also done: API run locally with real weights (SHA-256 starts 7d88ee05), website live with in-browser inference, live pipeline status, model card, Dependabot.

Written but never run with real data: notebook 02 (MLflow, real CSV validation, gate on real metrics, drift reference, NIH external validation).

Next:
1. Run notebook 02 on Colab; bring back candidate_metrics.json, drift_reference.json, the NIH report; update README/baseline/`MODEL_VERSION`.
2. Per-finding thresholds + calibration.
3. int8 browser model with the same parity gate.
