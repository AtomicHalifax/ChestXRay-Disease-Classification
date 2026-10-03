import json

import numpy as np
import pytest
from PIL import Image

from chexpert_cls.config import TARGET_DISEASES
from chexpert_cls.drift import build_reference
from chexpert_cls.drift_check import find_images, load_labels, load_nih_labels, summarize, to_markdown

K = len(TARGET_DISEASES)


def _ref(seed=0):
    return build_reference(np.random.default_rng(seed).beta(2, 5, (2000, K)))


def test_summarize_stable_vs_shift():
    rng = np.random.default_rng(1)
    names = [f"{i}.png" for i in range(300)]
    same = summarize(names, rng.beta(2, 5, (300, K)), [[]] * 300, _ref())
    shifted = summarize(names, rng.beta(6, 1.5, (300, K)), [[]] * 300, _ref())
    assert same["drift"]["status"] == "stable"
    assert shifted["drift"]["status"] == "major_shift"


def test_summarize_without_reference_and_quality_counts():
    r = summarize(["a.png", "b.png"], np.full((2, K), 0.3), [["image looks colour"], []])
    assert r["drift"]["status"] == "no_reference"
    assert r["input_quality"]["n_flagged"] == 1
    assert r["input_quality"]["flagged_fraction"] == 0.5


def test_external_validation_auroc():
    rng = np.random.default_rng(2)
    n = 200
    y = rng.integers(0, 2, (n, K))
    probs = np.clip(y * 0.6 + rng.normal(0.2, 0.15, (n, K)), 0, 1)  # informative scores
    names = [f"img{i}.png" for i in range(n)]
    labels = {nm: {d: int(y[i, j]) for j, d in enumerate(TARGET_DISEASES)} for i, nm in enumerate(names)}
    r = summarize(names, probs, [[]] * n, _ref(), labels)
    ev = r["external_validation"]
    assert ev["n_labelled"] == n
    assert all(v["auroc"] > 0.9 for v in ev["auroc"].values())
    assert "External validation" in to_markdown(r)


def test_auroc_skipped_when_only_one_class():
    names = ["a.png", "b.png", "c.png"]
    labels = {nm: {d: 0 for d in TARGET_DISEASES} for nm in names}
    r = summarize(names, np.full((3, K), 0.2), [[]] * 3, None, labels)
    assert all(v is None for v in r["external_validation"]["auroc"].values())
    assert r["external_validation"]["mean_auroc"] is None


def test_label_loaders(tmp_path):
    nih = tmp_path / "nih.csv"
    nih.write_text("Image Index,Finding Labels,Patient ID\n"
                   "00000001_000.png,Effusion|Edema,1\n00000002_000.png,No Finding,2\n")
    lab = load_nih_labels(nih)
    assert lab["00000001_000.png"]["Pleural Effusion"] == 1
    assert lab["00000001_000.png"]["Edema"] == 1
    assert lab["00000002_000.png"]["Cardiomegaly"] == 0

    gen = tmp_path / "labels.csv"
    gen.write_text("filename,Edema,Cardiomegaly\nx.png,1,0\n")
    assert load_labels(gen)["x.png"] == {"Cardiomegaly": 0, "Edema": 1}
    bad = tmp_path / "bad.csv"
    bad.write_text("filename,foo\nx.png,1\n")
    with pytest.raises(ValueError):
        load_labels(bad)


def test_find_images(tmp_path):
    (tmp_path / "sub").mkdir()
    for p in ["a.png", "sub/b.JPG", "notes.txt"]:
        (tmp_path / p).write_bytes(b"x")
    assert [p.name for p in find_images(tmp_path)] == ["a.png", "b.JPG"]


def test_cli_end_to_end(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import chexpert_cls.drift_check as dc
    import chexpert_cls.model as model_mod
    from chexpert_cls.model import build_model

    torch.manual_seed(0)
    monkeypatch.setattr(model_mod, "load_model", lambda *a, **k: build_model(pretrained=False).eval())
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8)).save(imgs / f"{i}.png")
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(imgs / "photo.png")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(_ref()))
    out = tmp_path / "out"
    code = dc.main([str(imgs), "--reference", str(ref), "--out", str(out)])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["n_images"] == 5
    assert report["input_quality"]["n_flagged"] == 1
    assert report["drift"]["status"] in {"stable", "moderate_shift", "major_shift"}
    assert (out / "predictions.csv").exists() and (out / "report.md").exists()
