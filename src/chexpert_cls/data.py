"""CheXpert label handling, patient-level splits and the PyTorch dataset."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TARGET_DISEASES

UNCERTAINTY_POLICIES = ("ignore", "ones", "zeros")

_PATIENT_RE = re.compile(r"(patient\d+)")


def patient_id(path: str) -> str:
    """Extract 'patientXXXXX' from a CheXpert image path."""
    m = _PATIENT_RE.search(path)
    if not m:
        raise ValueError(f"No patient id in path: {path}")
    return m.group(1)


def apply_label_policy(
    df: pd.DataFrame,
    policy: str = "ignore",
    diseases: list[str] = TARGET_DISEASES,
    frontal_only: bool = False,
) -> pd.DataFrame:
    """Map CheXpert labels {1, 0, -1, NaN} to binary targets.

    - NaN (not mentioned) -> 0
    - -1 (uncertain):
        ignore -> drop rows with any uncertain target  (what v1.0 used)
        ones   -> 1   (U-Ones, Irvin et al. 2019)
        zeros  -> 0   (U-Zeros)
    """
    if policy not in UNCERTAINTY_POLICIES:
        raise ValueError(f"policy must be one of {UNCERTAINTY_POLICIES}")
    out = df.copy()
    if frontal_only and "Frontal/Lateral" in out.columns:
        out = out[out["Frontal/Lateral"] == "Frontal"]
    out[diseases] = out[diseases].fillna(0)
    if policy == "ignore":
        out = out[~(out[diseases] == -1).any(axis=1)]
    elif policy == "ones":
        out[diseases] = out[diseases].replace(-1, 1)
    else:
        out[diseases] = out[diseases].replace(-1, 0)
    out[diseases] = out[diseases].astype("float32")
    return out.reset_index(drop=True)


def patient_split(
    df: pd.DataFrame, holdout_frac: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by patient so no patient appears in both parts.

    Used to carve a tuning set out of the training data for checkpoint
    selection and threshold tuning, so the official validation set is only
    touched once, for the final report.
    """
    pids = df["Path"].map(patient_id)
    unique = np.array(sorted(pids.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_hold = max(1, int(round(len(unique) * holdout_frac)))
    hold = set(unique[:n_hold])
    is_hold = pids.isin(hold)
    return df[~is_hold].reset_index(drop=True), df[is_hold].reset_index(drop=True)


def resolve_image_path(root: Path, csv_path: str) -> Path:
    """CheXpert CSVs store 'CheXpert-v1.0-small/train/...'. Accept roots that
    either contain that prefix folder or are the extracted folder itself."""
    root = Path(root)
    direct = root / csv_path
    if direct.exists():
        return direct
    stripped = re.sub(r"^CheXpert-v1\.0(-small)?/", "", csv_path)
    return root / stripped


def build_transforms(train: bool):
    from torchvision import transforms

    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class CheXpertDataset:
    """Minimal torch Dataset (duck-typed so this module imports without torch)."""

    def __init__(self, df: pd.DataFrame, root: Path, transform=None,
                 diseases: list[str] = TARGET_DISEASES):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.diseases = diseases

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        row = self.df.iloc[idx]
        img = Image.open(resolve_image_path(self.root, row["Path"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(row[self.diseases].to_numpy(dtype="float32"))
        return img, target
