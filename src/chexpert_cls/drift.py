"""Prediction-drift detection with the Population Stability Index (PSI).

Idea: on the validation set we know what the model's output distribution
looks like for each finding. In production we keep the last N predictions
and compare. A large shift means the incoming images differ from what the
model was validated on (new scanner, new hospital, wrong body part...).

PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.

Build the reference once, from evaluate.py's saved predictions:
    python -m chexpert_cls.drift --predictions results/valid_predictions.npz \
        --out models/drift_reference.json
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .config import TARGET_DISEASES

N_BINS = 10
EPS = 1e-4


def histogram(probs: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Fraction of probabilities in each of n_bins equal-width bins on [0, 1]."""
    counts, _ = np.histogram(np.clip(probs, 0, 1), bins=n_bins, range=(0.0, 1.0))
    return counts / max(len(probs), 1)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    """Population Stability Index between two binned distributions."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference(probs: np.ndarray, diseases: list[str] = TARGET_DISEASES) -> dict:
    """probs: [n_images, n_findings] validation-set probabilities."""
    return {
        "n_bins": N_BINS,
        "n_samples": int(len(probs)),
        "histograms": {d: histogram(probs[:, i]).tolist() for i, d in enumerate(diseases)},
    }


class DriftMonitor:
    """Thread-safe rolling window of predictions, compared to a reference."""

    def __init__(self, reference: dict, window: int = 500, min_samples: int = 50):
        self.reference = {d: np.asarray(h) for d, h in reference["histograms"].items()}
        self.n_bins = int(reference.get("n_bins", N_BINS))
        self.window = deque(maxlen=window)
        self.min_samples = min_samples
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "DriftMonitor":
        return cls(json.loads(Path(path).read_text()), **kw)

    def add(self, probs: dict[str, float]) -> None:
        with self._lock:
            self.window.append([probs[d] for d in TARGET_DISEASES])

    def scores(self) -> dict[str, float] | None:
        """PSI per finding, or None until min_samples predictions have arrived."""
        with self._lock:
            if len(self.window) < self.min_samples:
                return None
            arr = np.asarray(self.window)
        return {d: psi(self.reference[d], histogram(arr[:, i], self.n_bins))
                for i, d in enumerate(TARGET_DISEASES)}

    def status(self) -> dict:
        s = self.scores()
        worst = max(s.values()) if s else None
        level = ("insufficient_data" if s is None else
                 "major_shift" if worst > 0.25 else "moderate_shift" if worst > 0.1 else "stable")
        return {"status": level, "window_size": len(self.window), "min_samples": self.min_samples, "psi": s}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True, help=".npz from evaluate.py (keys y, p)")
    ap.add_argument("--out", type=Path, default=Path("models/drift_reference.json"))
    args = ap.parse_args(argv)
    ref = build_reference(np.load(args.predictions)["p"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ref, indent=2))
    print(f"wrote {args.out} from {ref['n_samples']} predictions")


if __name__ == "__main__":
    main()
