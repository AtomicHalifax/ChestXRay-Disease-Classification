"""Train DenseNet121 on CheXpert with a leak-free protocol.

Differences from the original notebook (v1.0):
  * the checkpoint is selected on a patient-level tuning split carved out
    of train.csv, NOT on the official validation set;
  * fixed seeds, mixed precision on GPU, frontal-only option;
  * the uncertainty policy is a flag (ignore / ones / zeros).

    python -m chexpert_cls.train --data-root /path/to/CheXpert-v1.0-small \
        --epochs 10 --policy ignore --out runs/densenet121
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TARGET_DISEASES
from .data import CheXpertDataset, apply_label_policy, build_transforms, patient_split
from .metrics import bootstrap_auroc
from .model import build_model
from .validation import validate_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device, non_blocking=True))).float().cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def mean_auroc(y: np.ndarray, p: np.ndarray) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import roc_auc_score

    per = {d: float(roc_auc_score(y[:, i], p[:, i])) for i, d in enumerate(TARGET_DISEASES)}
    return float(np.mean(list(per.values()))), per


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Folder containing train.csv, valid.csv, train/, valid/")
    ap.add_argument("--out", type=Path, default=Path("runs/densenet121"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--policy", choices=["ignore", "ones", "zeros"], default="ignore")
    ap.add_argument("--frontal-only", action="store_true")
    ap.add_argument("--tune-frac", type=float, default=0.05,
                    help="Fraction of training PATIENTS held out for checkpoint selection")
    ap.add_argument("--pos-weight", action="store_true", help="Class-balanced BCE")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow", action="store_true", help="Log params/metrics/artifacts to MLflow")
    ap.add_argument("--experiment", default="chexpert-densenet121")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_root / "train.csv")
    problems = validate_csv(df, min_rows=100)
    if problems:
        raise SystemExit("train.csv failed validation:\n  " + "\n  ".join(problems))
    df = apply_label_policy(df, args.policy, frontal_only=args.frontal_only)
    train_df, tune_df = patient_split(df, args.tune_frac, args.seed)
    print(f"train images {len(train_df):,} | tune images {len(tune_df):,} (patient-disjoint)")

    train_ds = CheXpertDataset(train_df, args.data_root, build_transforms(train=True))
    tune_ds = CheXpertDataset(tune_df, args.data_root, build_transforms(train=False))
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    tune_dl = DataLoader(tune_ds, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_model(pretrained=True).to(device)
    pos_weight = None
    if args.pos_weight:
        pos = train_df[TARGET_DISEASES].sum().to_numpy()
        pos_weight = torch.tensor((len(train_df) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    tracker = _Tracker(args) if args.mlflow else None
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        y, p = predict_loader(model, tune_dl, device)
        m, per = mean_auroc(y, p)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "tune_mean_auroc": m, **per})
        print(f"epoch {epoch:02d} loss {running / len(train_dl):.4f} tune mAUROC {m:.4f}")
        if tracker:
            tracker.epoch(history[-1])
        if m > best:
            best = m
            torch.save(model.state_dict(), args.out / "best.pth")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))

    # Final, single look at the official validation set.
    model.load_state_dict(torch.load(args.out / "best.pth", map_location=device, weights_only=True))
    valid = apply_label_policy(pd.read_csv(args.data_root / "valid.csv"), "zeros", frontal_only=args.frontal_only)
    vdl = DataLoader(CheXpertDataset(valid, args.data_root, build_transforms(False)), 64, num_workers=args.workers)
    y, p = predict_loader(model, vdl, device)
    report = {}
    for i, d in enumerate(TARGET_DISEASES):
        auc, lo, hi = bootstrap_auroc(y[:, i], p[:, i])
        report[d] = {"auroc": auc, "ci95": [lo, hi]}
        print(f"{d:<18} AUROC {auc:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    report["mean_auroc"] = float(np.mean([v["auroc"] for v in report.values()]))
    print(f"mean AUROC {report['mean_auroc']:.4f}")
    (args.out / "valid_report.json").write_text(json.dumps(report, indent=2))
    if tracker:
        tracker.finish(args.out, report)


class _Tracker:
    """Thin MLflow wrapper so mlflow stays an optional dependency."""

    def __init__(self, args):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_experiment(args.experiment)
        mlflow.start_run()
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})

    def epoch(self, row: dict) -> None:
        step = row["epoch"]
        self.mlflow.log_metrics({k.replace(" ", "_"): v for k, v in row.items() if k != "epoch"}, step=step)

    def finish(self, out: Path, report: dict) -> None:
        self.mlflow.log_metric("valid_mean_auroc", report["mean_auroc"])
        for d in TARGET_DISEASES:
            self.mlflow.log_metric(f"valid_auroc_{d.replace(' ', '_')}", report[d]["auroc"])
        for name in ("best.pth", "history.json", "valid_report.json"):
            self.mlflow.log_artifact(str(out / name))
        self.mlflow.end_run()


if __name__ == "__main__":
    main()
