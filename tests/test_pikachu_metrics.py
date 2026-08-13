"""Tests for telemetry shared by every registered Pikachu task."""

import pytest

from mjlab.tasks.pikachu_metrics import PIKACHU_JOINT_NAMES
from mjlab.tasks.registry import load_env_cfg

PIKACHU_TASK_IDS = (
  "Mjlab-Velocity-Rough-Pikachu",
  "Mjlab-Velocity-Flat-Pikachu",
  "Mjlab-Tracking-Flat-Pikachu-Down",
  "Mjlab-Tracking-Flat-Pikachu-Up",
)
JOINT_METRIC_PREFIXES = (
  "joint_position_rad/",
  "joint_velocity_rad_s/",
  "joint_torque_nm/",
)


@pytest.mark.parametrize("task_id", PIKACHU_TASK_IDS)
@pytest.mark.parametrize("play", [False, True])
def test_every_pikachu_task_exposes_joint_telemetry(
  task_id: str,
  play: bool,
) -> None:
  """Training and play configs should expose the same 14-joint telemetry."""
  cfg = load_env_cfg(task_id, play=play)

  assert "base_height_m" in cfg.metrics
  for prefix in JOINT_METRIC_PREFIXES:
    expected = {f"{prefix}{joint_name}" for joint_name in PIKACHU_JOINT_NAMES}
    actual = {name for name in cfg.metrics if name.startswith(prefix)}
    assert actual == expected
    assert all(cfg.metrics[name].reduce == "max_abs" for name in actual)
