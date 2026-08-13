"""Contracts for plot-log CSV discovery and standalone dashboards."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

discover_csv_files = runpy.run_path("scripts/plot_logs/serve.py")["discover_csv_files"]

METRICS_DASHBOARD = Path("scripts/plot_logs/metrics_csv_dashboard.html")
RESPONSE_DASHBOARD = Path("scripts/plot_logs/response_csv_dashboard.html")


def test_csv_discovery_filters_and_sorts_dashboard_inputs(tmp_path: Path) -> None:
  metrics_old = tmp_path / "logs" / "run a" / "metrics_old.csv"
  metrics_new = tmp_path / "logs" / "run_b" / "metrics_new.csv"
  response = tmp_path / "logs" / "response" / "task" / "run" / "response.csv"
  summary = response.with_name("summary.csv")
  for path in (metrics_old, metrics_new, response, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("header\n", encoding="utf-8")
  os.utime(metrics_old, (10.0, 10.0))
  os.utime(metrics_new, (20.0, 20.0))

  metrics = discover_csv_files(tmp_path, "metrics")
  responses = discover_csv_files(tmp_path, "response")

  assert [item["path"] for item in metrics] == [
    "logs/run_b/metrics_new.csv",
    "logs/run a/metrics_old.csv",
  ]
  assert metrics[1]["url"] == "/logs/run%20a/metrics_old.csv"
  assert [item["path"] for item in responses] == ["logs/response/task/run/response.csv"]


def test_csv_discovery_rejects_unknown_kind(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="Unsupported CSV kind"):
    discover_csv_files(tmp_path, "unknown")


def test_metrics_dashboard_auto_discovers_csv() -> None:
  content = METRICS_DASHBOARD.read_text(encoding="utf-8")

  assert 'id="csv-picker"' in content
  assert "/api/csv-files?kind=metrics" in content
  assert "discoverCsvs(true)" in content
  assert "loadCsvText" in content


def test_response_dashboard_includes_raw_sine_latex_and_nan_diagnostics() -> None:
  content = RESPONSE_DASHBOARD.read_text(encoding="utf-8")

  assert 'id="csv-picker"' in content
  assert "/api/csv-files?kind=response" in content
  assert 'id="sine-time-chart"' in content
  assert "function renderSineWaves" in content
  assert "tex-chtml.js" in content
  assert "\\begin{aligned}" in content
  assert "function renderNaNDiagnostics" in content
  assert "discoverCsvs(true)" in content
