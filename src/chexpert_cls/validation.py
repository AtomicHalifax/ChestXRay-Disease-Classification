"""Data-quality gates: run before training (CSV) and before inference (image)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_DISEASES

ALLOWED_LABELS = {1.0, 0.0, -1.0}
REQUIRED_COLUMNS = ["Path", *TARGET_DISEASES]


def validate_csv(df: pd.DataFrame, min_rows: int = 1) -> list[str]:
    """Return a list of problems with a CheXpert-style CSV (empty == OK)."""
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if len(df) < min_rows:
        errors.append(f"expected at least {min_rows} rows, got {len(df)}")
    if df["Path"].isna().any():
        errors.append(f"{int(df['Path'].isna().sum())} rows with empty Path")
    if df["Path"].duplicated().any():
        errors.append(f"{int(df['Path'].duplicated().sum())} duplicated image paths")
    bad_path = ~df["Path"].astype(str).str.contains(r"patient\d+/study\d+/", regex=True)
    if bad_path.any():
        errors.append(f"{int(bad_path.sum())} paths not in patientX/studyY/ layout")
    for d in TARGET_DISEASES:
        vals = set(df[d].dropna().unique().tolist())
        unexpected = vals - ALLOWED_LABELS
        if unexpected:
            errors.append(f"{d}: unexpected label values {sorted(unexpected)}")
    return errors


def image_warnings(img, min_side: int = 128) -> list[str]:
    """Cheap out-of-distribution checks for an uploaded image (PIL.Image).

    Chest radiographs are grayscale and roughly portrait/square. These checks
    do not prove an image is a chest X-ray; they catch obvious misuse
    (selfies, screenshots, tiny thumbnails) so the API can warn or reject.
    """
    warnings: list[str] = []
    w, h = img.size
    if min(w, h) < min_side:
        warnings.append(f"image is small ({w}x{h}); model was trained on >=320px films")
    ratio = w / h
    if not 0.5 <= ratio <= 2.0:
        warnings.append(f"unusual aspect ratio {ratio:.2f}")
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    # mean per-pixel channel spread: ~0 for grayscale X-rays, large for photos
    colorfulness = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    if colorfulness > 12.0:
        warnings.append(f"image looks colour (spread {colorfulness:.1f}); chest X-rays are grayscale")
    return warnings
