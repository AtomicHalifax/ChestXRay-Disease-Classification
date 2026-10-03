"""Model quality gate: refuse to promote a model that is worse than the current one.

    python -m chexpert_cls.gate --baseline models/baseline_metrics.json \
        --candidate runs/densenet121/valid_report.json

Exit code 0 = pass, 1 = fail. Writes a Markdown table to --summary (in CI,
point it at $GITHUB_STEP_SUMMARY so the result shows on the run page).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import TARGET_DISEASES


def load_metrics(path: str | Path) -> dict:
    """Normalise either report format to {"mean_auroc": float, "per_finding": {name: auroc}}.

    Accepts the baseline format ({"mean_auroc", "per_finding": {...}}) and the
    report written by train.py / evaluate.py ({finding: {"auroc", "ci95"}, "mean_auroc"}).
    """
    raw = json.loads(Path(path).read_text())
    if "per_finding" in raw:
        per = {k: float(v) for k, v in raw["per_finding"].items()}
    else:
        per = {d: float(raw[d]["auroc"]) for d in TARGET_DISEASES if d in raw}
    missing = [d for d in TARGET_DISEASES if d not in per]
    if missing:
        raise ValueError(f"{path}: missing findings {missing}")
    mean = float(raw.get("mean_auroc", sum(per.values()) / len(per)))
    return {"mean_auroc": mean, "per_finding": per, "model_version": raw.get("model_version")}


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    rows: list[tuple[str, float, float, float]] = field(default_factory=list)

    def markdown(self) -> str:
        lines = ["### Model quality gate: " + ("PASSED" if self.passed else "FAILED"), "",
                 "| Metric | Baseline | Candidate | Change |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {n} | {b:.4f} | {c:.4f} | {d:+.4f} |" for n, b, c, d in self.rows]
        if self.failures:
            lines += ["", *[f"- {f}" for f in self.failures]]
        return "\n".join(lines) + "\n"


def compare(baseline: dict, candidate: dict, max_mean_drop: float = 0.01,
            max_finding_drop: float = 0.03) -> GateResult:
    """Fail if mean AUROC drops by more than max_mean_drop, or any single
    finding drops by more than max_finding_drop (absolute AUROC points)."""
    res = GateResult(passed=True)
    d_mean = candidate["mean_auroc"] - baseline["mean_auroc"]
    res.rows.append(("Mean AUROC", baseline["mean_auroc"], candidate["mean_auroc"], d_mean))
    if d_mean < -max_mean_drop:
        res.failures.append(f"mean AUROC dropped {-d_mean:.4f} (limit {max_mean_drop})")
    for d in TARGET_DISEASES:
        b, c = baseline["per_finding"][d], candidate["per_finding"][d]
        res.rows.append((d, b, c, c - b))
        if c - b < -max_finding_drop:
            res.failures.append(f"{d} AUROC dropped {b - c:.4f} (limit {max_finding_drop})")
    res.passed = not res.failures
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, default=Path("models/baseline_metrics.json"))
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--max-mean-drop", type=float, default=0.01)
    ap.add_argument("--max-finding-drop", type=float, default=0.03)
    ap.add_argument("--summary", type=Path, default=None, help="Append the Markdown report here")
    args = ap.parse_args(argv)

    res = compare(load_metrics(args.baseline), load_metrics(args.candidate),
                  args.max_mean_drop, args.max_finding_drop)
    report = res.markdown()
    print(report)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(report)
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
