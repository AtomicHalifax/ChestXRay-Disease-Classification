"""Evaluation metrics that only need NumPy / scikit-learn.

The CheXpert validation set has just 234 images, so a single AUROC number
hides a lot of uncertainty. Everything here reports intervals or
calibration alongside the point estimate.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """AUROC with a percentile bootstrap (1 - alpha) confidence interval.

    Resamples that contain only one class are skipped.
    Returns (auroc, ci_low, ci_high).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    point = roc_auc_score(y_true, y_score)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        stats.append(roc_auc_score(yt, y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error with equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # rightmost bin includes 1.0
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1.

    Pick this on a tuning split, never on the split you report results on.
    """
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), y_score)
    best = int(np.argmax(tpr - fpr))
    # roc_curve's first threshold is +inf; clamp to a usable probability
    return float(min(thr[best], 1.0))


def sensitivity_specificity(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec
