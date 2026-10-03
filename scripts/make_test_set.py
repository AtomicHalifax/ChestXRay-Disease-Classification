"""Copy a small, labelled test set out of YOUR CheXpert validation split.

    python scripts/make_test_set.py --data-root /path/to/CheXpert-v1.0-small --n 10

Writes samples/chexpert_private/ (git-ignored) with the images and labels.csv,
choosing images so every finding has at least one positive example.
Then compare model output with the radiologist labels:

    python -m chexpert_cls.drift_check samples/chexpert_private --labels samples/chexpert_private/labels.csv

CheXpert's licence allows research use only. Do NOT commit these images or
put them on the public website.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chexpert_cls.config import TARGET_DISEASES  # noqa: E402
from chexpert_cls.data import apply_label_policy, resolve_image_path  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("samples/chexpert_private"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    df = apply_label_policy(pd.read_csv(args.data_root / "valid.csv"), "zeros", frontal_only=True)
    df = df.sample(frac=1, random_state=args.seed)
    picked = []
    for d in TARGET_DISEASES:  # one positive per finding first
        pos = df[(df[d] == 1) & (~df["Path"].isin(picked))]
        if len(pos):
            picked.append(pos.iloc[0]["Path"])
    normals = df[(df[TARGET_DISEASES].sum(axis=1) == 0) & (~df["Path"].isin(picked))]
    picked += normals["Path"].head(max(0, args.n - len(picked))).tolist()
    sel = df[df["Path"].isin(picked)].head(args.n)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in sel.iterrows():
        name = r["Path"].split("/valid/")[-1].replace("/", "_")
        shutil.copy(resolve_image_path(args.data_root, r["Path"]), args.out / name)
        rows.append({"filename": name, **{d: int(r[d]) for d in TARGET_DISEASES}})
    pd.DataFrame(rows).to_csv(args.out / "labels.csv", index=False)
    print(f"{len(rows)} images + labels.csv -> {args.out}  (private: do not commit)")


if __name__ == "__main__":
    main()
