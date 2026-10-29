"""Contracts for plot-log CSV discovery and standalone dashboards."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path("scripts/plot_logs/serve.py")
discover_csv_files = MODULE["discover_csv_files"]
bind_dashboard_server = MODULE["bind_dashboard_server"]

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


def test_csv_discovery_includes_experiment_prefix(tmp_path: Path) -> None:
  path = tmp_path / "logs" / "run" / "实验A_metrics_model_42.csv"
  path.parent.mkdir(parents=True)
  path.write_text("time_s,metric\n0,1\n", encoding="utf-8")
  found = discover_csv_files(tmp_path, "metrics")
  assert len(found) == 1 and found[0]["name"] == path.name


def test_nested_experiment_discovery_associates_media(tmp_path: Path) -> None:
  path = tmp_path / "logs/rsl_rl/task/run/experiment1/experiment1_metrics_m.csv"
  path.parent.mkdir(parents=True)
  path.write_text("time_s,metric\n0,1\n", encoding="utf-8")
  path.with_suffix(".mp4").write_bytes(b"video")
  path.with_suffix(".recording.json").write_text("{}", encoding="utf-8")
  found = discover_csv_files(tmp_path, "metrics")
  assert len(found) == 1
  assert (
    found[0]["video_url"]
    == "/" + path.with_suffix(".mp4").relative_to(tmp_path).as_posix()
  )
  assert found[0]["manifest_url"].endswith(".recording.json")
  assert found[0]["experiment"] == "experiment1"


def test_metrics_dashboard_auto_discovers_csv() -> None:
  content = METRICS_DASHBOARD.read_text(encoding="utf-8")

  assert 'id="csv-picker"' in content
  assert "/api/csv-files?kind=metrics" in content
  assert "discoverCsvs(true)" in content
  assert "loadCsvText" in content


def test_dashboard_bundles_offline_chart_library_and_notice() -> None:
  directory = METRICS_DASHBOARD.parent
  assert (directory / "echarts.min.js").is_file()
  assert 'version="5.5.0"' in (directory / "echarts.min.js").read_text()
  assert (
    "The Apache Software Foundation" in (directory / "ECHARTS_NOTICE.txt").read_text()
  )


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


def test_bind_dashboard_server_walks_past_busy_port() -> None:
  handler = MODULE["DashboardRequestHandler"]
  first = bind_dashboard_server("127.0.0.1", 0, handler)
  try:
    busy_port = first.server_address[1]

    with pytest.raises(OSError):
      bind_dashboard_server("127.0.0.1", busy_port, handler, strict=True)

    second = bind_dashboard_server("127.0.0.1", busy_port, handler, max_tries=5)
    try:
      assert second.server_address[1] != busy_port
    finally:
      second.server_close()
  finally:
    first.server_close()


def test_bind_dashboard_server_rejects_exhausted_port_range() -> None:
  handler = MODULE["DashboardRequestHandler"]
  first = bind_dashboard_server("127.0.0.1", 0, handler)
  try:
    with pytest.raises(OSError, match="均不可用"):
      bind_dashboard_server("127.0.0.1", first.server_address[1], handler, max_tries=1)
  finally:
    first.server_close()
