"""Static contract tests for the standalone Metrics CSV dashboard."""

from pathlib import Path

DASHBOARD = Path("scripts/plot_logs/metrics_csv_dashboard.html")


def test_metrics_csv_dashboard_contains_core_analysis_features() -> None:
  """The self-contained page should preserve its CSV/debug/export contract."""
  content = DASHBOARD.read_text(encoding="utf-8")

  assert 'id="file-input"' in content
  assert "function parseCsv" in content
  assert "joint_position_rad/" in content
  assert "joint_velocity_rad_s/" in content
  assert "joint_torque_nm/" in content
  assert (
    "const CANONICAL_JOINT_ORDER = ['hip_pitch','hip_roll','hip_yaw','knee'" in content
  )
  assert "function saveOverviewPng" in content
  assert "function savePairsPng" in content
  assert "function saveSelectedCsv" in content
  assert "1920,3780" in content
  assert "RMS" in content
  assert "P95" in content
