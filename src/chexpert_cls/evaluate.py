"""Evaluate a checkpoint on the CheXpert validation set with uncertainty.

Reports per-disease AUROC with 95% bootstrap CIs, average precision,
expected calibration error, and sensitivity/specificity at 0.5.

    python -m chexpert_cls.evaluate --data-root /path/to/CheXpert-v1.0-small \
        [--weights models/best_densenet121.pth] [--out results/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from .config import TARGET_DISEASES
from .data import CheXpertDataset, apply_label_policy, build_transforms
from .metrics import bootstrap_auroc, expected_calibration_error, sensitivity_specificity
from .model import load_model
from .train import predict_loader


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(TARGET_DISEASES):
        auc, lo, hi = bootstrap_auroc(y[:, i], p[:, i])
        sens, spec = sensitivity_specificity(y[:, i], p[:, i], threshold)
        rows.append({
            "disease": d,
            "n_pos": int(y[:, i].sum()),
            "auroc": round(auc, 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "avg_precision": round(float(average_precision_score(y[:, i], p[:, i])), 4),
            "ece": round(expected_calibration_error(y[:, i], p[:, i]), 4),
            f"sens@{threshold}": round(sens, 4),
            f"spec@{threshold}": round(spec, 4),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--frontal-only", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    # the validation set is radiologist-labelled and has no -1 labels
    valid = apply_label_policy(pd.read_csv(args.data_root / "valid.csv"), "zeros", frontal_only=args.frontal_only)
    dl = DataLoader(CheXpertDataset(valid, args.data_root, build_transforms(False)), batch_size=64)
    y, p = predict_loader(model, dl, device)

    table = evaluate(y, p)
    print(table.to_string(index=False))
    print(f"\nmean AUROC {table['auroc'].mean():.4f}  (n={len(y)} images)")

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "valid_metrics.csv", index=False)
    np.savez(args.out / "valid_predictions.npz", y=y, p=p)
    (args.out / "summary.json").write_text(json.dumps(
        {"mean_auroc": float(table["auroc"].mean()), "n_images": int(len(y))}, indent=2))


if __name__ == "__main__":
    main()
