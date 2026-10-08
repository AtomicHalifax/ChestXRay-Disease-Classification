"""FastAPI inference service.

    uvicorn chexpert_cls.api:app --host 0.0.0.0 --port 8000

Environment variables
  CHEXPERT_WEIGHTS          local checkpoint path (default: download from HF Hub)
  CHEXPERT_WEIGHTS_SHA256   if set, refuse to start unless the checkpoint matches
  CHEXPERT_RANDOM_WEIGHTS   "1" = random weights, for tests / smoke checks only
  CHEXPERT_STRICT_INPUT     "1" = reject (422) images that fail the OOD checks
  CHEXPERT_MAX_UPLOAD_MB    upload size limit (default 10); only PNG/JPEG accepted
  CHEXPERT_RATE_LIMIT       POST requests per minute per client IP (default 30)
  CHEXPERT_CORS_ORIGINS     comma-separated allowed origins (default: the project site + localhost)
  CHEXPERT_DRIFT_REFERENCE  drift reference JSON (default models/drift_reference.json;
                            drift monitoring is off if the file does not exist)
  CHEXPERT_DRIFT_WINDOW     number of recent predictions compared (default 500)
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from . import __version__
from .config import MODEL_VERSION, TARGET_DISEASES
from .drift import DriftMonitor
from .model import build_model, load_model, resolve_weights
from .predict import gradcam_image, predict_image
from .validation import image_warnings

REQUESTS = Counter("chexpert_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("chexpert_latency_seconds", "Inference latency", ["endpoint"])
# Prediction-distribution histograms: the raw signal for drift monitoring.
PROBS = Histogram("chexpert_probability", "Predicted probability", ["finding"],
                  buckets=[i / 10 for i in range(1, 11)])
DRIFT = Gauge("chexpert_drift_psi", "PSI of recent predictions vs validation reference", ["finding"])

STATE: dict = {}
Image.MAX_IMAGE_PIXELS = 50_000_000  # PIL raises DecompressionBombError above 2x this
ORIGINS = os.getenv("CHEXPERT_CORS_ORIGINS",
                    "https://atomichalifax.github.io,http://localhost:8000,http://localhost:7860").split(",")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.getenv("CHEXPERT_RANDOM_WEIGHTS") == "1":
        torch.manual_seed(0)
        STATE.update(model=build_model(pretrained=False).eval().to(device), sha256="random", device=device)
        return
    path = resolve_weights(os.getenv("CHEXPERT_WEIGHTS"))
    digest = _sha256(path)
    expected = os.getenv("CHEXPERT_WEIGHTS_SHA256")
    if expected and expected.lower() != digest:
        raise RuntimeError(f"checkpoint sha256 {digest} != expected {expected}")
    STATE.update(model=load_model(path, device), sha256=digest, device=device)


def _load_drift() -> None:
    ref = Path(os.getenv("CHEXPERT_DRIFT_REFERENCE", "models/drift_reference.json"))
    if ref.exists():
        STATE["drift"] = DriftMonitor.from_file(ref, window=int(os.getenv("CHEXPERT_DRIFT_WINDOW", "500")))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    _load_drift()
    yield
    STATE.clear()


app = FastAPI(title="CheXpert DenseNet121 API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"])

# ponytail: in-memory fixed window per IP; one process only, and behind a proxy every client shares one IP.
# Use the proxy's limiter (or slowapi + Redis) if this is ever deployed publicly.
_HITS: dict = {}
_WINDOW = [0]


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.method == "POST":
        w = int(time.time() // 60)
        if _WINDOW[0] != w:
            _HITS.clear()
            _WINDOW[0] = w
        ip = request.client.host if request.client else "?"
        _HITS[ip] = _HITS.get(ip, 0) + 1
        if _HITS[ip] > int(os.getenv("CHEXPERT_RATE_LIMIT", "30")):
            return JSONResponse({"detail": "too many requests"}, 429, headers={"Retry-After": "60"})
    return await call_next(request)


def _read_image(upload: UploadFile) -> Image.Image:
    limit = int(float(os.getenv("CHEXPERT_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    data = upload.file.read(limit + 1)  # never read more than the limit into memory
    if len(data) > limit:
        raise HTTPException(413, "file too large")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ("PNG", "JPEG", "MPO"):  # MPO = JPEG from many phone cameras
            raise HTTPException(415, "only PNG or JPEG images are accepted")
        img.load()
    except Image.DecompressionBombError:
        raise HTTPException(413, "image dimensions too large")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "could not decode the file as an image")
    return img


@app.get("/health")
def health():
    return {"status": "ok" if "model" in STATE else "loading"}


@app.get("/version")
def version():
    return {"api": __version__, "model": MODEL_VERSION, "weights_sha256": STATE.get("sha256"),
            "findings": TARGET_DISEASES, "device": STATE.get("device")}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    try:
        img = _read_image(file)
        warnings = image_warnings(img)
        if warnings and os.getenv("CHEXPERT_STRICT_INPUT") == "1":
            raise HTTPException(422, {"error": "input failed quality checks", "warnings": warnings})
        probs = predict_image(STATE["model"], img, STATE["device"])
    except HTTPException as e:
        REQUESTS.labels("predict", str(e.status_code)).inc()
        raise
    for k, v in probs.items():
        PROBS.labels(k).observe(v)
    monitor = STATE.get("drift")
    if monitor is not None:
        monitor.add(probs)
        for k, v in (monitor.scores() or {}).items():
            DRIFT.labels(k).set(v)
    dt = time.perf_counter() - t0
    LATENCY.labels("predict").observe(dt)
    REQUESTS.labels("predict", "200").inc()
    return {
        "model": MODEL_VERSION,
        "findings": probs,
        "top_finding": max(probs, key=probs.get),
        "warnings": warnings,
        "latency_ms": round(dt * 1000, 1),
        "disclaimer": "Research use only. Not a medical device.",
    }


@app.post("/explain", responses={200: {"content": {"image/png": {}}}})
def explain(file: UploadFile = File(...), finding: str | None = Query(None)):
    t0 = time.perf_counter()
    try:
        img = _read_image(file)
        if finding is not None and finding not in TARGET_DISEASES:
            raise HTTPException(422, f"finding must be one of {TARGET_DISEASES}")
        if finding is None:
            probs = predict_image(STATE["model"], img, STATE["device"])
            finding = max(probs, key=probs.get)
        overlay = gradcam_image(STATE["model"], img, finding, STATE["device"])
    except HTTPException as e:
        REQUESTS.labels("explain", str(e.status_code)).inc()
        raise
    buf = io.BytesIO()
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype("uint8")).save(buf, format="PNG")
    LATENCY.labels("explain").observe(time.perf_counter() - t0)
    REQUESTS.labels("explain", "200").inc()
    return Response(buf.getvalue(), media_type="image/png", headers={"X-Finding": finding})


@app.get("/drift")
def drift():
    monitor = STATE.get("drift")
    if monitor is None:
        return {"status": "disabled", "reason": "no drift reference loaded"}
    return monitor.status()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
