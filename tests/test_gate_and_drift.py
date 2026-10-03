import json

import numpy as np
import pytest

from chexpert_cls.config import TARGET_DISEASES
from chexpert_cls.drift import DriftMonitor, build_reference, histogram, psi
from chexpert_cls.gate import compare, load_metrics, main as gate_main

BASE = {"mean_auroc": 0.88, "per_finding": {d: 0.88 for d in TARGET_DISEASES}}


def _cand(**overrides):
    per = {d: 0.88 for d in TARGET_DISEASES}
    per.update(overrides)
    return {"mean_auroc": sum(per.values()) / len(per), "per_finding": per}


# ---------- quality gate ----------

def test_gate_passes_equal_or_better():
    assert compare(BASE, _cand()).passed
    assert compare(BASE, _cand(Edema=0.95)).passed


def test_gate_fails_on_single_finding_drop():
    res = compare(BASE, _cand(Edema=0.84))  # -0.04 > 0.03 limit
    assert not res.passed
    assert any("Edema" in f for f in res.failures)


def test_gate_fails_on_mean_drop():
    worse = {d: 0.86 for d in TARGET_DISEASES}  # each -0.02 (ok), mean -0.02 (fail)
    res = compare(BASE, {"mean_auroc": 0.86, "per_finding": worse})
    assert not res.passed
    assert any("mean" in f for f in res.failures)


def test_load_metrics_accepts_both_formats(tmp_path):
    base = tmp_path / "b.json"
    base.write_text(json.dumps(BASE))
    report = tmp_path / "r.json"
    report.write_text(json.dumps({**{d: {"auroc": 0.9, "ci95": [0.85, 0.95]} for d in TARGET_DISEASES},
                                  "mean_auroc": 0.9}))
    assert load_metrics(base)["per_finding"]["Edema"] == 0.88
    assert load_metrics(report)["per_finding"]["Edema"] == 0.9
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mean_auroc": 0.9, "per_finding": {"Edema": 0.9}}))
    with pytest.raises(ValueError):
        load_metrics(bad)


def test_gate_cli_exit_code_and_summary(tmp_path):
    base, cand, summary = tmp_path / "b.json", tmp_path / "c.json", tmp_path / "s.md"
    base.write_text(json.dumps(BASE))
    cand.write_text(json.dumps(_cand(Cardiomegaly=0.70)))
    assert gate_main(["--baseline", str(base), "--candidate", str(cand), "--summary", str(summary)]) == 1
    assert "FAILED" in summary.read_text()


def test_committed_baseline_is_valid():
    m = load_metrics("models/baseline_metrics.json")
    assert m["mean_auroc"] == pytest.approx(np.mean(list(m["per_finding"].values())), abs=1e-3)


# ---------- drift ----------

def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.beta(2, 5, 5000)
    b = rng.beta(2, 5, 5000)
    shifted = rng.beta(5, 2, 5000)
    assert psi(histogram(a), histogram(b)) < 0.02
    assert psi(histogram(a), histogram(shifted)) > 0.25


def _reference(rng):
    return build_reference(rng.beta(2, 5, (2000, len(TARGET_DISEASES))))


def test_monitor_waits_for_min_samples_then_reports():
    rng = np.random.default_rng(1)
    mon = DriftMonitor(_reference(rng), window=300, min_samples=50)
    assert mon.status()["status"] == "insufficient_data"
    for row in rng.beta(2, 5, (300, len(TARGET_DISEASES))):
        mon.add(dict(zip(TARGET_DISEASES, row)))
    assert mon.status()["status"] == "stable"


def test_monitor_flags_major_shift():
    rng = np.random.default_rng(2)
    mon = DriftMonitor(_reference(rng), window=300, min_samples=50)
    for row in rng.beta(6, 1.5, (300, len(TARGET_DISEASES))):  # model suddenly says "yes" to everything
        mon.add(dict(zip(TARGET_DISEASES, row)))
    st = mon.status()
    assert st["status"] == "major_shift"
    assert all(v > 0.25 for v in st["psi"].values())


def test_reference_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(_reference(rng)))
    mon = DriftMonitor.from_file(path, min_samples=1)
    mon.add({d: 0.1 for d in TARGET_DISEASES})
    assert set(mon.scores()) == set(TARGET_DISEASES)
