"""Batch drift check: does a new folder of X-rays look like what the model was validated on?

Use it when someone hands you a new dataset (another hospital, a new scanner,
a public dataset such as NIH ChestX-ray14) *before* trusting the model on it.

    python -m chexpert_cls.drift_check path/to/new_images/ --out drift_report/

What it reports
  1. Input quality: how many images fail the grayscale / size / aspect checks.
  2. Prediction drift: PSI per finding vs models/drift_reference.json
     (stable < 0.1 <= moderate < 0.25 <= major).
  3. Optional external validation: if you pass labels, AUROC per finding with
     95% bootstrap CIs, i.e. "how well does the model do on this new data?"
       --labels labels.csv          columns: filename + one 0/1 column per finding
       --nih-labels Data_Entry_2017.csv   NIH ChestX-ray14 format ("Image Index",
                                    "Finding Labels" like "Effusion|Edema")

Exit code: 0 normally; 2 if --fail-on-shift and any finding shows a major shift
(so it can gate an automated pipeline).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TARGET_DISEASES
from .drift import histogram, psi

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
NIH_NAME_MAP = {"Effusion": "Pleural Effusion"}  # NIH label -> our finding name


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_labels(path: Path) -> dict[str, dict[str, int]]:
    """Generic labels CSV: a 'filename' (or 'Path') column + 0/1 finding columns."""
    df = pd.read_csv(path)
    key = "filename" if "filename" in df.columns else "Path"
    if key not in df.columns:
        raise ValueError(f"{path}: needs a 'filename' or 'Path' column")
    present = [d for d in TARGET_DISEASES if d in df.columns]
    if not present:
        raise ValueError(f"{path}: no finding columns among {TARGET_DISEASES}")
    out = {}
    for _, row in df.iterrows():
        out[Path(str(row[key])).name] = {d: int(row[d] == 1) for d in present}
    return out


def load_nih_labels(path: Path) -> dict[str, dict[str, int]]:
    """NIH ChestX-ray14 'Data_Entry_2017.csv' / Kaggle 'sample_labels.csv'."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            found = {NIH_NAME_MAP.get(x.strip(), x.strip()) for x in row["Finding Labels"].split("|")}
            out[row["Image Index"]] = {d: int(d in found) for d in TARGET_DISEASES}
    return out


def summarize(names: list[str], probs: np.ndarray, warnings: list[list[str]],
              reference: dict | None = None, labels: dict[str, dict[str, int]] | None = None) -> dict:
    """Pure function (no torch): build the report from predictions."""
    from .metrics import bootstrap_auroc

    n = len(names)
    report: dict = {
        "n_images": n,
        "input_quality": {
            "n_flagged": int(sum(1 for w in warnings if w)),
            "flagged_fraction": float(np.mean([bool(w) for w in warnings])) if n else 0.0,
            "examples": [{"file": nm, "warnings": w} for nm, w in zip(names, warnings) if w][:10],
        },
        "mean_probability": {d: float(probs[:, i].mean()) for i, d in enumerate(TARGET_DISEASES)} if n else {},
    }

    if reference is not None and n:
        per = {d: psi(reference["histograms"][d], histogram(probs[:, i], reference.get("n_bins", 10)))
               for i, d in enumerate(TARGET_DISEASES)}
        worst = max(per.values())
        report["drift"] = {
            "psi": per,
            "status": "major_shift" if worst > 0.25 else "moderate_shift" if worst > 0.1 else "stable",
            "reference_samples": reference.get("n_samples"),
            "note": "PSI on fewer than ~100 images is noisy" if n < 100 else None,
        }
    else:
        report["drift"] = {"status": "no_reference"}

    if labels:
        idx = [i for i, nm in enumerate(names) if nm in labels]
        ext = {"n_labelled": len(idx), "auroc": {}}
        for j, d in enumerate(TARGET_DISEASES):
            y = np.array([labels[names[i]].get(d, -1) for i in idx])
            keep = y >= 0
            if keep.sum() == 0 or len(set(y[keep])) < 2:
                ext["auroc"][d] = None  # need both positives and negatives
                continue
            auc, lo, hi = bootstrap_auroc(y[keep], probs[idx][keep, j], n_boot=1000)
            ext["auroc"][d] = {"auroc": auc, "ci95": [lo, hi], "n_pos": int(y[keep].sum()), "n": int(keep.sum())}
        vals = [v["auroc"] for v in ext["auroc"].values() if v]
        ext["mean_auroc"] = float(np.mean(vals)) if vals else None
        report["external_validation"] = ext
    return report


def to_markdown(r: dict) -> str:
    lines = [f"## Drift check: {r['n_images']} images", ""]
    q = r["input_quality"]
    lines.append(f"**Input quality:** {q['n_flagged']} flagged ({q['flagged_fraction']:.0%}).")
    d = r["drift"]
    lines += ["", f"**Prediction drift:** `{d['status']}`"]
    if "psi" in d:
        lines += ["", "| Finding | Mean prob | PSI |", "| --- | ---: | ---: |"]
        lines += [f"| {k} | {r['mean_probability'][k]:.3f} | {v:.3f} |" for k, v in d["psi"].items()]
        if d.get("note"):
            lines += ["", f"_{d['note']}_"]
    ev = r.get("external_validation")
    if ev:
        lines += ["", f"**External validation** on {ev['n_labelled']} labelled images:", "",
                  "| Finding | AUROC | 95% CI | positives |", "| --- | ---: | --- | ---: |"]
        for k, v in ev["auroc"].items():
            lines.append(f"| {k} | n/a | needs both classes | |" if v is None else
                         f"| {k} | {v['auroc']:.3f} | {v['ci95'][0]:.3f}–{v['ci95'][1]:.3f} | {v['n_pos']}/{v['n']} |")
        if ev.get("mean_auroc") is not None:
            lines += ["", f"Mean AUROC: **{ev['mean_auroc']:.3f}**"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--reference", type=Path, default=Path("models/drift_reference.json"))
    ap.add_argument("--weights", type=Path, default=None)
    labels = ap.add_mutually_exclusive_group()
    labels.add_argument("--labels", type=Path, default=None)
    labels.add_argument("--nih-labels", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    ap.add_argument("--out", type=Path, default=Path("drift_report"))
    ap.add_argument("--fail-on-shift", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from PIL import Image

    from .model import load_model
    from .predict import predict_image
    from .validation import image_warnings

    files = find_images(args.folder)[: args.limit]
    if not files:
        raise SystemExit(f"no images found in {args.folder}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    names, probs, warns = [], [], []
    for i, f in enumerate(files, 1):
        img = Image.open(f)
        warns.append(image_warnings(img))
        p = predict_image(model, img, device)
        probs.append([p[d] for d in TARGET_DISEASES])
        names.append(f.name)
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)
    probs = np.asarray(probs)

    reference = json.loads(args.reference.read_text()) if args.reference.exists() else None
    lab = load_labels(args.labels) if args.labels else load_nih_labels(args.nih_labels) if args.nih_labels else None
    report = summarize(names, probs, warns, reference, lab)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.out / "report.md").write_text(md)
    pd.DataFrame(probs, columns=TARGET_DISEASES).assign(filename=names).to_csv(args.out / "predictions.csv", index=False)
    print(md)
    if reference is None:
        print(f"(no drift reference at {args.reference}; build it with python -m chexpert_cls.drift)", file=sys.stderr)
    return 2 if args.fail_on_shift and report["drift"]["status"] == "major_shift" else 0


if __name__ == "__main__":
    sys.exit(main())
