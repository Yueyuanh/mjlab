"""Extraction of joint, actuator, and motor metadata from the task model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  ACTUATOR_J3507,
  ACTUATOR_J4310,
  GEARS_J3507,
  GEARS_J4310,
  ROTOR_INERTIAS_J3507,
  ROTOR_INERTIAS_J4310,
)
from mjlab.entity import Entity


@dataclass(frozen=True)
class JointModel:
  """Parameters needed to interpret one rotary actuator's response."""

  joint: str
  motor_model: str
  actuator_type: str
  parameter_status: str
  lower_limit_rad: float
  upper_limit_rad: float
  home_position_rad: float
  kp_nm_per_rad: float
  kd_nms_per_rad: float
  torque_limit_nm: float
  velocity_limit_rad_s: float
  nominal_voltage_v: float
  continuous_torque_nm: float
  torque_constant_nm_a: float
  back_emf_v_s_rad: float
  phase_resistance_ohm: float
  thermal_resistance_k_w: float
  motor_efficiency: float
  armature_kg_m2: float
  home_inertia_kg_m2: float
  link_inertia_at_home_kg_m2: float
  passive_damping_nms_per_rad: float
  frictionloss_nm: float
  gear_ratio_total: float
  rotor_inertia_0_kg_m2: float
  rotor_inertia_1_kg_m2: float
  rotor_inertia_2_kg_m2: float
  command_delay_min_steps: int
  command_delay_max_steps: int

  def as_dict(self) -> dict[str, str | float | int]:
    return asdict(self)


def _pikachu_motor(
  joint: str,
) -> tuple[
  str,
  float,
  float,
  tuple[float, float, float],
]:
  if joint.endswith(("_hip_pitch_joint", "_knee_joint")):
    return (
      "DM-J4310P-2EC-V1.1",
      ACTUATOR_J4310.velocity_limit,
      float(np.prod(GEARS_J4310)),
      ROTOR_INERTIAS_J4310,
    )
  return (
    "DM-J3507-2EC-3R",
    ACTUATOR_J3507.velocity_limit,
    float(np.prod(GEARS_J3507)),
    ROTOR_INERTIAS_J3507,
  )


def _home_mass_diagonal(model: mujoco.MjModel) -> np.ndarray:
  data = mujoco.MjData(model)
  if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  dense = np.empty((model.nv, model.nv), dtype=float)
  mujoco.mj_fullM(model, data, dense)
  return np.diag(dense)


def extract_joint_models(
  robot: Entity,
  model: mujoco.MjModel,
  task_id: str,
) -> dict[str, JointModel]:
  """Extract authoritative compiled values and known datasheet metadata."""
  mass_diag = _home_mass_diagonal(model)
  home = robot.data.default_joint_pos[0].cpu().numpy()
  limits = robot.data.joint_pos_limits[0].cpu().numpy()
  models: dict[str, JointModel] = {}

  for local_id, joint in enumerate(robot.joint_names):
    actuator = next(
      (candidate for candidate in robot.actuators if joint in candidate.target_names),
      None,
    )
    if actuator is None:
      continue
    cfg = actuator.cfg
    if not hasattr(cfg, "stiffness") or not hasattr(cfg, "damping"):
      continue

    joint_id = model.joint(joint).id
    dof_id = int(model.jnt_dofadr[joint_id])
    armature = float(model.dof_armature[dof_id])
    total_inertia = float(mass_diag[dof_id])
    effort = getattr(cfg, "effort_limit", None)
    torque_limit = float(effort) if effort is not None else float("inf")
    if "Pikachu" in task_id:
      motor, speed_limit, gear, rotor = _pikachu_motor(joint)
      status = "Rotor inertias and gear ratios contain placeholder reference values."
    else:
      motor = "unspecified"
      status = "No motor datasheet mapping is registered for this task."
      speed_limit = float("nan")
      gear = float("nan")
      rotor = (float("nan"),) * 3

    models[joint] = JointModel(
      joint=joint,
      motor_model=motor,
      actuator_type=type(cfg).__name__,
      parameter_status=status,
      nominal_voltage_v=float("nan"),
      continuous_torque_nm=float("nan"),
      torque_constant_nm_a=float("nan"),
      back_emf_v_s_rad=float("nan"),
      phase_resistance_ohm=float("nan"),
      thermal_resistance_k_w=float("nan"),
      motor_efficiency=float("nan"),
      lower_limit_rad=float(limits[local_id, 0]),
      upper_limit_rad=float(limits[local_id, 1]),
      home_position_rad=float(home[local_id]),
      kp_nm_per_rad=float(cfg.stiffness),
      kd_nms_per_rad=float(cfg.damping),
      torque_limit_nm=torque_limit,
      velocity_limit_rad_s=speed_limit,
      armature_kg_m2=armature,
      home_inertia_kg_m2=total_inertia,
      link_inertia_at_home_kg_m2=max(0.0, total_inertia - armature),
      passive_damping_nms_per_rad=float(model.dof_damping[dof_id]),
      frictionloss_nm=float(model.dof_frictionloss[dof_id]),
      gear_ratio_total=gear,
      rotor_inertia_0_kg_m2=float(rotor[0]),
      rotor_inertia_1_kg_m2=float(rotor[1]),
      rotor_inertia_2_kg_m2=float(rotor[2]),
      command_delay_min_steps=int(getattr(cfg, "delay_min_lag", 0)),
      command_delay_max_steps=int(getattr(cfg, "delay_max_lag", 0)),
    )
  return models


def model_predictions(joint: JointModel) -> dict[str, float]:
  """Return the local linear second-order prediction at the home pose."""
  inertia = joint.home_inertia_kg_m2
  kp = joint.kp_nm_per_rad
  total_damping = joint.kd_nms_per_rad + joint.passive_damping_nms_per_rad
  if inertia <= 0.0 or kp <= 0.0:
    return {}
  natural_frequency = float(np.sqrt(kp / inertia))
  damping_ratio = total_damping / (2.0 * np.sqrt(kp * inertia))
  settling = (
    4.0 / (damping_ratio * natural_frequency) if damping_ratio > 0.0 else float("inf")
  )
  overshoot = 0.0
  if 0.0 < damping_ratio < 1.0:
    overshoot = 100.0 * np.exp(-np.pi * damping_ratio / np.sqrt(1.0 - damping_ratio**2))
  return {
    "model_natural_frequency_rad_s": natural_frequency,
    "model_natural_frequency_hz": natural_frequency / (2.0 * np.pi),
    "model_damping_ratio": damping_ratio,
    "model_settling_time_2pct_s": float(settling),
    "model_overshoot_pct": float(overshoot),
  }
