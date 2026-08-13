"""Tests for limit-safe response protocols and dynamic analysis."""

import math

import mujoco
import pytest

from mjlab.entity import EntityCfg
from mjlab.response.analysis import analyze_sine_case, analyze_step_case
from mjlab.response.config import ResponseConfig
from mjlab.response.protocols import build_joint_cases
from mjlab.response.runner import _json_safe, _suspended_robot_cfg


def _row(
  *,
  response_type: str,
  time_s: float,
  command: float,
  measured: float,
  velocity: float = 0.0,
  torque: float = 0.0,
  frequency: float = 0.0,
  warmup_cycles: float = 0.0,
) -> dict[str, object]:
  return {
    "response_type": response_type,
    "case_id": f"joint__{response_type}",
    "joint": "joint",
    "motor_model": "test-motor",
    "frequency_hz": frequency,
    "direction": 1,
    "amplitude_rad": 1.0,
    "test_time_s": time_s,
    "stimulus_start_s": 0.2,
    "warmup_cycles": warmup_cycles,
    "command_pos_rad": command,
    "measured_pos_rad": measured,
    "measured_vel_rad_s": velocity,
    "actuator_torque_nm": torque,
    "mechanical_power_w": torque * velocity,
    "torque_saturated": 0,
  }


def test_joint_cases_stay_inside_margin():
  cfg = ResponseConfig(
    mode="all",
    limit_margin_fraction=0.1,
    step_amplitude_fraction=1.0,
    sine_amplitude_fraction=1.0,
    sine_frequencies_hz=(1.0,),
  )
  cases = build_joint_cases("joint", 0.2, -1.0, 2.0, cfg)

  assert len(cases) == 3
  for case in cases:
    for sample in range(101):
      command, _ = case.command(case.duration_s * sample / 100)
      assert -0.7 <= command <= 1.7


def test_step_metrics_detect_rise_and_overshoot():
  rows = []
  dt = 0.01
  for index in range(121):
    time_s = index * dt
    if time_s < 0.2:
      command = 0.0
      measured = 0.0
    else:
      command = 1.0
      elapsed = time_s - 0.2
      measured = min(1.1, elapsed / 0.2 * 1.1)
      if elapsed > 0.3:
        measured = 1.0
    rows.append(
      _row(
        response_type="step",
        time_s=time_s,
        command=command,
        measured=measured,
        velocity=0.0,
        torque=0.5,
      )
    )

  summary = analyze_step_case(rows)

  assert summary["rise_time_10_90_s"] == pytest.approx(0.14545, abs=0.012)
  assert summary["overshoot_pct"] == pytest.approx(10.0, abs=0.1)
  assert summary["settling_time_2pct_s"] == pytest.approx(0.31, abs=0.02)
  assert summary["steady_state_error_rad"] == pytest.approx(0.0)


def test_sine_metrics_recover_gain_and_phase():
  frequency = 2.0
  phase = math.radians(-30.0)
  rows = []
  for index in range(401):
    time_s = index * 0.005
    stimulus_time = max(0.0, time_s - 0.2)
    command = math.sin(2.0 * math.pi * frequency * stimulus_time)
    measured = 0.5 * math.sin(2.0 * math.pi * frequency * stimulus_time + phase)
    rows.append(
      _row(
        response_type="sine",
        time_s=time_s,
        command=command,
        measured=measured,
        frequency=frequency,
        warmup_cycles=1.0,
      )
    )

  summary = analyze_sine_case(rows)

  assert summary["gain_ratio"] == pytest.approx(0.5, rel=1e-3)
  assert summary["gain_db"] == pytest.approx(-6.0206, abs=1e-3)
  assert summary["phase_deg"] == pytest.approx(-30.0, abs=0.1)
  assert summary["response_fit_r2"] == pytest.approx(1.0, abs=1e-9)
  assert summary["thd_pct"] == pytest.approx(0.0, abs=1e-8)


def test_suspended_cfg_removes_free_joint_and_contact():
  xml = """
  <mujoco>
    <worldbody>
      <body name="base">
        <freejoint name="root"/>
        <geom name="body_geom" type="sphere" size="0.1"
              contype="1" conaffinity="1"/>
        <body name="link">
          <joint name="joint" type="hinge" range="-1 1"/>
          <geom name="link_geom" type="capsule" size="0.02 0.1"
                contype="2" conaffinity="3"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
  suspended = _suspended_robot_cfg(cfg, 1.25)
  spec = suspended.spec_fn()

  assert not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in spec.joints)
  assert all(geom.contype == 0 and geom.conaffinity == 0 for geom in spec.geoms)
  assert suspended.init_state.pos == (0.0, 0.0, 1.25)


def test_json_safe_replaces_nonfinite_values():
  value = {"nan": float("nan"), "inf": float("inf"), "items": (1.0,)}
  assert _json_safe(value) == {"nan": None, "inf": None, "items": [1.0]}
