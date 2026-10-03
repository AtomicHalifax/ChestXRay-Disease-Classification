"""Every metric used by the dashboard and alert rules must exist in the API."""

import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("torch")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]


def _api_metric_names() -> set[str]:
    import chexpert_cls.api  # noqa: F401  (registers the metrics)
    from prometheus_client import REGISTRY

    names = set()
    for fam in REGISTRY.collect():
        if not fam.name.startswith("chexpert"):
            continue
        if fam.type == "counter":
            names.add(fam.name + "_total")
        elif fam.type == "histogram":
            names |= {fam.name + s for s in ("_bucket", "_sum", "_count")}
        else:
            names.add(fam.name)
    return names


def _used_metric_names() -> set[str]:
    exprs = []
    alerts = yaml.safe_load((ROOT / "monitoring/prometheus/alerts.yml").read_text())
    exprs += [r["expr"] for g in alerts["groups"] for r in g["rules"]]
    dash = json.loads((ROOT / "monitoring/grafana/dashboards/chexpert.json").read_text())
    exprs += [t["expr"] for p in dash["panels"] for t in p.get("targets", [])]
    return {m for e in exprs for m in re.findall(r"\bchexpert_[a-z_]+\b", e)}


def test_dashboard_and_alerts_only_use_real_metrics():
    used, real = _used_metric_names(), _api_metric_names()
    assert used, "no metrics found in configs"
    assert used <= real, f"unknown metrics: {sorted(used - real)}"


def test_prometheus_scrapes_the_api_service():
    cfg = yaml.safe_load((ROOT / "monitoring/prometheus/prometheus.yml").read_text())
    targets = [t for job in cfg["scrape_configs"] for sc in job["static_configs"] for t in sc["targets"]]
    assert "api:8000" in targets
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert {"api", "prometheus", "grafana", "alertmanager"} <= set(compose["services"])
    am_targets = [t for a in cfg["alerting"]["alertmanagers"] for sc in a["static_configs"] for t in sc["targets"]]
    assert "alertmanager:9093" in am_targets
