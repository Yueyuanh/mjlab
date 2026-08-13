"""Tests for Viser metric CSV recording and loading."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mjlab.viewer.viser.term_plotter import (
  MetricsCsvRecorder,
  load_metrics_csv,
  save_metrics_curve_images,
)


def test_metrics_csv_recorder_roundtrip(tmp_path: Path) -> None:
  """Selected terms, simulation time, environment, and network round-trip."""
  current_network = ["Mjlab-Tracking-Flat-Pikachu-Up__model_499"]
  recorder = MetricsCsvRecorder(
    tmp_path / "selected-model-folder",
    lambda: current_network[0],
    step_dt=0.02,
  )

  path = recorder.start(["joint_position_rad/left_knee_joint", "base_height_m"])
  recorder.append(
    {
      "joint_position_rad/left_knee_joint": 0.25,
      "base_height_m": 0.18,
      "not_selected": 99.0,
    },
    env_idx=3,
    sim_step=100,
  )
  current_network[0] = "Mjlab-Tracking-Flat-Pikachu-Up__model_999"
  recorder.append(
    {
      "joint_position_rad/left_knee_joint": -0.5,
      "base_height_m": 0.17,
    },
    env_idx=4,
    sim_step=102,
  )
  assert recorder.stop() == path

  assert path.parent == tmp_path / "selected-model-folder"
  assert "Mjlab-Tracking-Flat-Pikachu-Up__model_499" in path.name
  content = path.read_bytes()
  assert b"not_selected" not in content

  loaded = load_metrics_csv(
    content,
    ["joint_position_rad/left_knee_joint", "base_height_m", "not_selected"],
  )
  np.testing.assert_allclose(loaded.time_s, [0.0, 0.04])
  np.testing.assert_allclose(
    loaded.values["joint_position_rad/left_knee_joint"],
    [0.25, -0.5],
  )
  np.testing.assert_allclose(loaded.values["base_height_m"], [0.18, 0.17])
  assert "not_selected" not in loaded.values
  assert loaded.network == "Mjlab-Tracking-Flat-Pikachu-Up__model_499"
  assert "model_999" in content.decode()


@pytest.mark.parametrize(
  ("content", "message"),
  [
    (b"value\n1\n", "time_s"),
    (b"time_s,unknown\n0,1\n", "no metric columns"),
    (b"time_s,metric\n1,2\n0,3\n", "non-decreasing"),
    (b"time_s,metric\n0,nope\n", "non-numeric"),
  ],
)
def test_load_metrics_csv_rejects_invalid_input(
  content: bytes,
  message: str,
) -> None:
  """Malformed or incompatible recordings should produce actionable errors."""
  with pytest.raises(ValueError, match=message):
    load_metrics_csv(content, ["metric"])


def test_save_metrics_curve_images_creates_1080p_png(tmp_path: Path) -> None:
  """Joint export should group sides and create overview/pair comparisons."""
  left_term = "joint_torque_nm/left_knee_joint"
  right_term = "joint_torque_nm/right_knee_joint"
  content = (
    f"time_s,network,{left_term},{right_term},base_height_m\n"
    "0.00,model_499,-1.5,1.4,0.18\n"
    "0.02,model_499,2.25,-2.1,0.17\n"
  ).encode()
  loaded = load_metrics_csv(content, [left_term, right_term, "base_height_m"])

  paths = save_metrics_curve_images(
    loaded,
    [left_term, right_term],
    csv_parent=tmp_path,
    csv_name="metrics_model_499.csv",
    colors={left_term: "#2563eb", right_term: "#dc2626"},
  )

  root = tmp_path / "images" / "metrics_model_499" / "joint_torque"
  expected_sizes = {
    root / "left" / "knee.png": (1920, 1080),
    root / "right" / "knee.png": (1920, 1080),
    root / "left" / "left_joints_comparison.png": (1920, 1080),
    root / "right" / "right_joints_comparison.png": (1920, 1080),
    root / "all_joints_comparison.png": (1920, 1080),
    root / "left_right_comparison_1x7.png": (1920, 3780),
  }
  assert set(paths) == set(expected_sizes)
  for path, size in expected_sizes.items():
    with Image.open(path) as image:
      assert image.format == "PNG"
      assert image.size == size
