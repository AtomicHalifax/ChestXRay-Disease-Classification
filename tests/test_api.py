"""API contract tests. Random weights, no network."""

import io
import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
os.environ["CHEXPERT_RANDOM_WEIGHTS"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from chexpert_cls.api import app  # noqa: E402
from chexpert_cls.config import TARGET_DISEASES  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan (model load)
        yield c


def _png(color=False, size=(320, 320)) -> bytes:
    rng = np.random.default_rng(0)
    if color:
        arr = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, size, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_version(client):
    assert client.get("/health").json() == {"status": "ok"}
    v = client.get("/version").json()
    assert v["findings"] == TARGET_DISEASES
    assert v["weights_sha256"] == "random"


def test_predict_contract(client):
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["findings"]) == set(TARGET_DISEASES)
    assert all(0 <= p <= 1 for p in body["findings"].values())
    assert body["top_finding"] in TARGET_DISEASES
    assert body["warnings"] == []


def test_predict_warns_on_colour_photo(client):
    r = client.post("/predict", files={"file": ("x.png", _png(color=True), "image/png")})
    assert r.status_code == 200
    assert any("colour" in w for w in r.json()["warnings"])


def test_strict_mode_rejects_bad_input(client, monkeypatch):
    monkeypatch.setenv("CHEXPERT_STRICT_INPUT", "1")
    r = client.post("/predict", files={"file": ("x.png", _png(color=True), "image/png")})
    assert r.status_code == 422


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_rejects_large_upload(client, monkeypatch):
    monkeypatch.setenv("CHEXPERT_MAX_UPLOAD_MB", "0.0001")
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 413


def test_explain_returns_png(client):
    r = client.post("/explain?finding=Edema", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-finding"] == "Edema"
    assert Image.open(io.BytesIO(r.content)).size == (224, 224)


def test_explain_rejects_unknown_finding(client):
    r = client.post("/explain?finding=Fracture", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 422


def test_drift_disabled_without_reference(client):
    assert client.get("/drift").json()["status"] == "disabled"


def test_drift_endpoint_and_gauge(client):
    from chexpert_cls.api import STATE
    from chexpert_cls.drift import DriftMonitor, build_reference

    ref = build_reference(np.random.default_rng(0).beta(2, 5, (500, len(TARGET_DISEASES))))
    STATE["drift"] = DriftMonitor(ref, window=10, min_samples=3)
    try:
        for _ in range(3):
            client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
        body = client.get("/drift").json()
        assert body["status"] in {"stable", "moderate_shift", "major_shift"}
        assert set(body["psi"]) == set(TARGET_DISEASES)
        assert "chexpert_drift_psi" in client.get("/metrics").text
    finally:
        STATE.pop("drift", None)


def test_metrics_exposed(client):
    client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    text = client.get("/metrics").text
    assert "chexpert_requests_total" in text
    assert "chexpert_probability_bucket" in text
