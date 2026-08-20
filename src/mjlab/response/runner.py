"""Suspended MJWarp actuator-response runner and CSV export."""

from __future__ import annotations

import csv
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.response.analysis import analyze_response_rows
from mjlab.response.config import ResponseConfig
from mjlab.response.model import JointModel, extract_joint_models
from mjlab.response.protocols import ResponseCase, build_joint_cases
from mjlab.sim import Simulation
from mjlab.tasks.registry import load_env_cfg

_SCHEMA_VERSION = "mjlab-response-v1"


@dataclass(frozen=True)
class ResponseRunResult:
  """Paths and counts produced by a completed response run."""

  output_dir: Path
  response_csv: Path
  summary_csv: Path
  metadata_json: Path
  num_cases: int
  num_samples: int


def _suspended_robot_cfg(robot_cfg: EntityCfg, height_m: float) -> EntityCfg:
  """Return a fixed-base, contact-disabled copy of a task robot."""
  cfg = deepcopy(robot_cfg)
  original_spec_fn = cfg.spec_fn

  def build_suspended_spec() -> mujoco.MjSpec:
    spec = original_spec_fn()
    free_joints = [
      joint for joint in tuple(spec.joints) if joint.type == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joints) > 1:
      raise ValueError("Response runner supports at most one robot free joint.")
    for joint in free_joints:
      spec.delete(joint)
    for geom in spec.geoms:
      geom.contype = 0
      geom.conaffinity = 0
    return spec

  cfg.spec_fn = build_suspended_spec
  cfg.init_state.pos = (0.0, 0.0, height_m)
  cfg.init_state.lin_vel = (0.0, 0.0, 0.0)
  cfg.init_state.ang_vel = (0.0, 0.0, 0.0)
  return cfg


def _build_runner(
  task_id: str,
  cfg: ResponseConfig,
) -> tuple[Entity, Simulation, dict[str, JointModel], float]:
  env_cfg = load_env_cfg(task_id, play=True)
  if "robot" not in env_cfg.scene.entities:
    raise ValueError(f"Task {task_id!r} has no scene entity named 'robot'.")
  robot_cfg = _suspended_robot_cfg(
    env_cfg.scene.entities["robot"], cfg.suspend_height_m
  )
  robot = Entity(robot_cfg)
  model = robot.compile()
  sim = Simulation(
    num_envs=1,
    cfg=env_cfg.sim,
    model=model,
    device=cfg.device,
  )
  robot.initialize(model, sim.model, sim.data, cfg.device)
  joint_models = extract_joint_models(robot, model, task_id)
  return robot, sim, joint_models, float(model.opt.timestep)


def _reset_case(
  robot: Entity,
  sim: Simulation,
  joint_index: int,
  center_rad: float,
) -> torch.Tensor:
  sim.reset()
  robot.reset()
  positions = robot.data.default_joint_pos.clone()
  positions[:, joint_index] = center_rad
  velocities = torch.zeros_like(robot.data.default_joint_vel)
  robot.write_joint_state_to_sim(positions, velocities)
  robot.set_joint_position_target(positions)
  robot.set_joint_velocity_target(velocities)
  robot.set_joint_effort_target(velocities)
  robot.write_data_to_sim()
  sim.forward()
  return positions


def _contact_count(sim: Simulation) -> int:
  ncon = getattr(sim.data, "ncon", None)
  if ncon is None:
    return 0
  return int(ncon[0].item())


def _sample_row(
  *,
  task_id: str,
  run_id: str,
  case: ResponseCase,
  joint_model: JointModel,
  sample: int,
  sim_step: int,
  time_s: float,
  test_time_s: float,
  dt: float,
  joint_index: int,
  robot: Entity,
  sim: Simulation,
  target: torch.Tensor,
  command_pos: float,
  command_vel: float,
  energy_increment: float,
  suspend_height_m: float,
) -> dict[str, Any]:
  positions = robot.data.joint_pos[0].detach().cpu().numpy()
  velocities = robot.data.joint_vel[0].detach().cpu().numpy()
  torques = robot.data.qfrc_actuator[0].detach().cpu().numpy()
  position = float(positions[joint_index])
  velocity = float(velocities[joint_index])
  torque = float(torques[joint_index])
  error = command_pos - position
  raw_pd_torque = (
    joint_model.kp_nm_per_rad * error - joint_model.kd_nms_per_rad * velocity
  )
  torque_limit = joint_model.torque_limit_nm
  clipped_pd_torque = float(np.clip(raw_pd_torque, -torque_limit, torque_limit))
  power = torque * velocity
  other_error = positions - target[0].detach().cpu().numpy()
  other_error = np.delete(other_error, joint_index)
  contact_count = _contact_count(sim)
  finite_speed_limit = (
    joint_model.velocity_limit_rad_s
    if np.isfinite(joint_model.velocity_limit_rad_s)
    else float("nan")
  )
  return {
    "schema_version": _SCHEMA_VERSION,
    "run_id": run_id,
    "task": task_id,
    "backend": "mujoco-warp",
    "sample": sample,
    "sim_step": sim_step,
    "time_s": time_s,
    "test_time_s": test_time_s,
    "stimulus_start_s": case.pre_duration_s,
    "stimulus_time_s": max(0.0, test_time_s - case.pre_duration_s),
    "response_type": case.response_type,
    "case_id": case.case_id,
    "joint": case.joint,
    "direction": case.direction,
    "frequency_hz": case.frequency_hz,
    "warmup_cycles": case.warmup_cycles,
    "amplitude_rad": case.amplitude_rad,
    "command_pos_rad": command_pos,
    "command_vel_rad_s": command_vel,
    "measured_pos_rad": position,
    "measured_vel_rad_s": velocity,
    "position_error_rad": error,
    "actuator_torque_nm": torque,
    "unclamped_pd_torque_nm": raw_pd_torque,
    "clipped_pd_torque_nm": clipped_pd_torque,
    "torque_saturated": int(
      np.isfinite(torque_limit)
      and torque_limit > 0.0
      and abs(torque) >= 0.995 * torque_limit
    ),
    "torque_utilization": (
      abs(torque) / torque_limit
      if np.isfinite(torque_limit) and torque_limit > 0.0
      else float("nan")
    ),
    "speed_utilization": (
      abs(velocity) / finite_speed_limit
      if np.isfinite(finite_speed_limit) and finite_speed_limit > 0.0
      else float("nan")
    ),
    "mechanical_power_w": power,
    "absolute_energy_j_increment": energy_increment,
    "position_limit_margin_rad": min(
      position - joint_model.lower_limit_rad,
      joint_model.upper_limit_rad - position,
    ),
    "other_joint_error_rms_rad": (
      float(np.sqrt(np.mean(other_error**2))) if len(other_error) else 0.0
    ),
    "contact_count": contact_count,
    "suspended_height_m": suspend_height_m,
    "physics_dt_s": dt,
    "motor_model": joint_model.motor_model,
    "actuator_type": joint_model.actuator_type,
    "parameter_status": joint_model.parameter_status,
    "nominal_voltage_v": joint_model.nominal_voltage_v,
    "continuous_torque_nm": joint_model.continuous_torque_nm,
    "torque_constant_nm_a": joint_model.torque_constant_nm_a,
    "back_emf_v_s_rad": joint_model.back_emf_v_s_rad,
    "phase_resistance_ohm": joint_model.phase_resistance_ohm,
    "thermal_resistance_k_w": joint_model.thermal_resistance_k_w,
    "motor_efficiency": joint_model.motor_efficiency,
    "kp_nm_per_rad": joint_model.kp_nm_per_rad,
    "kd_nms_per_rad": joint_model.kd_nms_per_rad,
    "torque_limit_nm": joint_model.torque_limit_nm,
    "velocity_limit_rad_s": joint_model.velocity_limit_rad_s,
    "joint_limit_lower_rad": joint_model.lower_limit_rad,
    "joint_limit_upper_rad": joint_model.upper_limit_rad,
    "home_position_rad": joint_model.home_position_rad,
    "armature_kg_m2": joint_model.armature_kg_m2,
    "home_inertia_kg_m2": joint_model.home_inertia_kg_m2,
    "link_inertia_at_home_kg_m2": joint_model.link_inertia_at_home_kg_m2,
    "passive_damping_nms_per_rad": joint_model.passive_damping_nms_per_rad,
    "frictionloss_nm": joint_model.frictionloss_nm,
    "gear_ratio_total": joint_model.gear_ratio_total,
    "rotor_inertia_kg_m2": joint_model.rotor_inertia_kg_m2,
    "command_delay_min_steps": joint_model.command_delay_min_steps,
    "command_delay_max_steps": joint_model.command_delay_max_steps,
  }


def _run_case(
  *,
  task_id: str,
  run_id: str,
  case: ResponseCase,
  joint_model: JointModel,
  robot: Entity,
  sim: Simulation,
  dt: float,
  cfg: ResponseConfig,
  global_sample: int,
  global_step: int,
  global_time: float,
) -> tuple[list[dict[str, Any]], int, int, float]:
  joint_index = robot.joint_names.index(case.joint)
  target = _reset_case(robot, sim, joint_index, case.center_rad)
  num_steps = max(1, math.ceil(case.duration_s / dt))
  rows: list[dict[str, Any]] = []
  energy_since_sample = 0.0

  for case_step in range(num_steps):
    command_time = case_step * dt
    command_pos, command_vel = case.command(command_time)
    target[:, joint_index] = command_pos
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)

    velocity = float(robot.data.joint_vel[0, joint_index].item())
    torque = float(robot.data.qfrc_actuator[0, joint_index].item())
    energy_since_sample += abs(velocity * torque) * dt
    global_step += 1
    global_time += dt
    test_time = (case_step + 1) * dt

    if case_step % cfg.sample_stride != 0 and case_step != num_steps - 1:
      continue
    row = _sample_row(
      task_id=task_id,
      run_id=run_id,
      case=case,
      joint_model=joint_model,
      sample=global_sample,
      sim_step=global_step,
      time_s=global_time,
      test_time_s=test_time,
      dt=dt,
      joint_index=joint_index,
      robot=robot,
      sim=sim,
      target=target,
      command_pos=command_pos,
      command_vel=command_vel,
      energy_increment=energy_since_sample,
      suspend_height_m=cfg.suspend_height_m,
    )
    if cfg.strict_no_contact and int(row["contact_count"]) != 0:
      raise RuntimeError(
        f"Contact detected in suspended case {case.case_id}; aborting response run."
      )
    rows.append(row)
    global_sample += 1
    energy_since_sample = 0.0
  return rows, global_sample, global_step, global_time


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    raise ValueError(f"Cannot write empty response CSV: {path}")
  fields: list[str] = []
  seen: set[str] = set()
  for row in rows:
    for key in row:
      if key not in seen:
        seen.add(key)
        fields.append(key)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _json_safe(value: Any) -> Any:
  """Convert non-finite floats to JSON null recursively."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return value


def _safe_task_name(task_id: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-") or "task"


def run_response(task_id: str, cfg: ResponseConfig) -> ResponseRunResult:
  """Run every selected joint/case and export raw, summary, and metadata files."""
  cfg.validate()
  robot, sim, joint_models, dt = _build_runner(task_id, cfg)
  pattern = re.compile(cfg.joint_pattern)
  selected = [
    joint
    for joint in robot.joint_names
    if joint in joint_models
    and pattern.fullmatch(joint)
    and model_joint_is_hinge(sim.mj_model, joint)
  ]
  if not selected:
    raise ValueError(
      f"joint_pattern {cfg.joint_pattern!r} matched no position-controlled hinge joints."
    )

  cases: list[ResponseCase] = []
  for joint in selected:
    model = joint_models[joint]
    cases.extend(
      build_joint_cases(
        joint,
        model.home_position_rad,
        model.lower_limit_rad,
        model.upper_limit_rad,
        cfg,
      )
    )
  if not cases:
    raise ValueError("No safe response cases could be generated from the joint limits.")

  now = datetime.now(timezone.utc)
  run_id = now.strftime("%Y%m%dT%H%M%S_%fZ")
  output_dir = (
    Path(cfg.output_root).expanduser().resolve() / _safe_task_name(task_id) / run_id
  )
  output_dir.mkdir(parents=True, exist_ok=False)
  response_csv = output_dir / "response.csv"
  summary_csv = output_dir / "summary.csv"
  metadata_json = output_dir / "metadata.json"

  simulated_seconds = sum(case.duration_s for case in cases)
  print(
    f"[INFO] Suspended response run: {len(selected)} joints, {len(cases)} cases, "
    f"{simulated_seconds:.1f} simulated seconds at dt={dt:g}s."
  )
  print("[INFO] Terrain absent, root fixed, all geom collisions disabled.")

  rows: list[dict[str, Any]] = []
  global_sample = 0
  global_step = 0
  global_time = 0.0
  for index, case in enumerate(cases, start=1):
    print(f"[{index:03d}/{len(cases):03d}] {case.case_id}")
    case_rows, global_sample, global_step, global_time = _run_case(
      task_id=task_id,
      run_id=run_id,
      case=case,
      joint_model=joint_models[case.joint],
      robot=robot,
      sim=sim,
      dt=dt,
      cfg=cfg,
      global_sample=global_sample,
      global_step=global_step,
      global_time=global_time,
    )
    rows.extend(case_rows)

  summaries = analyze_response_rows(rows, joint_models, cfg)
  _write_csv(response_csv, rows)
  _write_csv(summary_csv, summaries)
  metadata = {
    "schema_version": _SCHEMA_VERSION,
    "run_id": run_id,
    "created_utc": now.isoformat(),
    "task": task_id,
    "backend": "mujoco-warp",
    "device": cfg.device,
    "physics_dt_s": dt,
    "suspension": {
      "root_fixed": True,
      "height_m": cfg.suspend_height_m,
      "terrain_present": False,
      "collisions_enabled": False,
      "contact_invariant": "contact_count == 0",
    },
    "config": asdict(cfg),
    "joints": [joint_models[joint].as_dict() for joint in selected],
    "case_ids": [case.case_id for case in cases],
    "files": {
      "raw": response_csv.name,
      "summary": summary_csv.name,
    },
  }
  metadata_json.write_text(
    json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"[INFO] Raw response CSV: {response_csv}")
  print(f"[INFO] Analysis summary: {summary_csv}")
  print(f"[INFO] Motor metadata: {metadata_json}")
  return ResponseRunResult(
    output_dir=output_dir,
    response_csv=response_csv,
    summary_csv=summary_csv,
    metadata_json=metadata_json,
    num_cases=len(cases),
    num_samples=len(rows),
  )


def model_joint_is_hinge(model: mujoco.MjModel, joint: str) -> bool:
  """Return whether a compiled joint uses rotary hinge coordinates."""
  joint_id = model.joint(joint).id
  return model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
