#!/usr/bin/env sh
# Fire a fake alert straight into Alertmanager to check notifications end to end.
# Usage: scripts/send_test_alert.sh [alertmanager_url]
AM="${1:-http://localhost:9093}"
curl -sf -XPOST "$AM/api/v2/alerts" -H 'Content-Type: application/json' -d '[{
  "labels": {"alertname": "ChexpertTestAlert", "severity": "info"},
  "annotations": {"summary": "Test alert from the CheXpert monitoring stack. If you can read this, notifications work."}
}]' && echo "test alert sent to $AM (check the Alertmanager UI or your inbox in ~30-60 s)"
