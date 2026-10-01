"""Fast tests that need only NumPy / pandas / scikit-learn."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from chexpert_cls.config import TARGET_DISEASES
from chexpert_cls.data import apply_label_policy, patient_id, patient_split, resolve_image_path
from chexpert_cls.metrics import (
    bootstrap_auroc,
    expected_calibration_error,
    sensitivity_specificity,
    youden_threshold,
)


def _fake_df(n_patients=40, studies=3):
    rng = np.random.default_rng(0)
    rows = []
    for p in range(n_patients):
        for s in range(studies):
            row = {"Path": f"CheXpert-v1.0-small/train/patient{p:05d}/study{s + 1}/view1_frontal.jpg",
                   "Frontal/Lateral": "Frontal" if s % 2 == 0 else "Lateral"}
            row.update({d: rng.choice([1.0, 0.0, -1.0, np.nan]) for d in TARGET_DISEASES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_patient_id():
    assert patient_id("CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg") == "patient64541"
    with pytest.raises(ValueError):
        patient_id("no/patient/here.jpg")


def test_patient_split_is_disjoint_and_complete():
    df = _fake_df()
    tr, ho = patient_split(df, holdout_frac=0.2, seed=1)
    assert len(tr) + len(ho) == len(df)
    assert set(tr.Path.map(patient_id)).isdisjoint(set(ho.Path.map(patient_id)))
    assert ho.Path.map(patient_id).nunique() == 8


@pytest.mark.parametrize("policy", ["ignore", "ones", "zeros"])
def test_label_policies_produce_binary_targets(policy):
    out = apply_label_policy(_fake_df(), policy)
    vals = set(np.unique(out[TARGET_DISEASES].to_numpy()))
    assert vals <= {0.0, 1.0}
    if policy == "ignore":
        assert len(out) < len(_fake_df())
    else:
        assert len(out) == len(_fake_df())


def test_frontal_only():
    out = apply_label_policy(_fake_df(), "zeros", frontal_only=True)
    assert (out["Frontal/Lateral"] == "Frontal").all()


def test_resolve_image_path_strips_prefix(tmp_path):
    p = resolve_image_path(tmp_path, "CheXpert-v1.0-small/valid/patient1/study1/view1_frontal.jpg")
    assert p == tmp_path / "valid/patient1/study1/view1_frontal.jpg"


def test_bootstrap_auroc_matches_point_and_brackets_it():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y * 0.8 + rng.normal(0, 0.6, 300)
    auc, lo, hi = bootstrap_auroc(y, s, n_boot=500)
    assert auc == pytest.approx(roc_auc_score(y, s))
    assert lo < auc < hi


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)
    assert expected_calibration_error(y, 1 - y.astype(float)) == pytest.approx(1.0)


def test_threshold_and_sens_spec():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(y, s)
    assert 0.3 < t <= 0.7
    assert sensitivity_specificity(y, s, t) == (1.0, 1.0)
