import numpy as np
import pandas as pd
from PIL import Image

from chexpert_cls.config import TARGET_DISEASES
from chexpert_cls.validation import image_warnings, validate_csv


def _df(n=5):
    return pd.DataFrame({
        "Path": [f"CheXpert-v1.0-small/train/patient{i:05d}/study1/view1_frontal.jpg" for i in range(n)],
        **{d: [1.0, 0.0, -1.0, np.nan, 1.0][:n] for d in TARGET_DISEASES},
    })


def test_valid_csv_passes():
    assert validate_csv(_df()) == []


def test_missing_column():
    assert "missing columns" in validate_csv(_df().drop(columns=["Edema"]))[0]


def test_bad_labels_and_duplicates_and_paths():
    df = _df()
    df.loc[0, "Edema"] = 2.0
    df.loc[1, "Path"] = df.loc[2, "Path"]
    df.loc[3, "Path"] = "somewhere/else.jpg"
    errs = " | ".join(validate_csv(df))
    assert "unexpected label values [2.0]" in errs
    assert "duplicated" in errs
    assert "layout" in errs


def test_min_rows():
    assert any("at least" in e for e in validate_csv(_df(), min_rows=100))


def test_image_warnings():
    gray = Image.fromarray(np.full((320, 320), 128, dtype=np.uint8))
    assert image_warnings(gray) == []
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (320, 320, 3), dtype=np.uint8))
    assert any("colour" in w for w in image_warnings(photo))
    tiny = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    assert any("small" in w for w in image_warnings(tiny))
    wide = Image.fromarray(np.zeros((200, 900), dtype=np.uint8))
    assert any("aspect" in w for w in image_warnings(wide))
