"""Shared telemetry metrics for all Pikachu tasks.

在 velocity / tracking 任务上做两件体检：**(a) 支撑相脚底打滑**（与离线
``scripts/check_reference_feasibility.py`` 的滑脚口径一致，训练中就能看到曲线），
**(b) 电机规格占用**（峰值扭矩/角速度相对 DM 电机 spec 的比值，>1.0 即该步超
规格）+ **动作平滑度**（action 二阶差分）。三者共同回答"这个策略能不能上实机"：
打滑说明步态在拖脚，超规格说明电机跟不上，高频抖动说明策略在抽搐。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  ACTUATOR_J3507,
  ACTUATOR_J4310,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.utils.actuator import ElectricActuator

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PIKACHU_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
)

# 足部 site 名（Pikachu 资产内置，挂在 ankle body 下），与 velocity 任务的
# SITE_NAMES 一致，用来量支撑相脚速。
_FOOT_SITE_NAMES = ("left_foot", "right_foot")

# 脚底接地接触传感器名（velocity 任务已定义同名 sensor，这里复用；tracking 任务
# 没有则补一个）。primary 为左右踝 body 子树（含脚掌胶囊），secondary 为地形。
_FEET_CONTACT_SENSOR = "feet_ground_contact"

# 电机规格（关节名正则 -> ElectricActuator），数值单一来源 pikachu_constants.py：
#   J4310（髋 pitch/膝）velocity_limit=30 rad/s, effort_limit=10 N·m
#   J3507（髋 roll/yaw/踝/臂）velocity_limit=30/3=10, effort_limit=5*3=15
_ACTUATOR_GROUPS: tuple[tuple[str, ElectricActuator], ...] = (
  (r".*(hip_pitch|knee)_joint", ACTUATOR_J4310),
  (r".*(hip_roll|hip_yaw|ankle|arm_pitch|arm_roll)_joint", ACTUATOR_J3507),
)


def _actuator_spec(joint_name: str) -> tuple[float, float]:
  """Return ``(velocity_limit, effort_limit)`` for a Pikachu joint."""
  for pattern, actuator in _ACTUATOR_GROUPS:
    if re.fullmatch(pattern, joint_name):
      return actuator.velocity_limit, actuator.effort_limit
  msg = f"No actuator spec for joint {joint_name}"
  raise ValueError(msg)


def _contact_foot_speed(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return ``(in_contact, foot_xy_speed)`` for the two foot sites.

  ``in_contact``: [B, N] float mask (found > 0) per foot site.
  ``foot_xy_speed``: [B, N] horizontal world speed of each foot site.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  foot_speed = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  return in_contact, foot_speed


def foot_slip_velocity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """支撑相水平脚速均值（m/s）：只统计当前接触的脚。

  与离线检查脚本 ``stance_slip_mean`` 同口径。接触脚全部静止时为 0；
  越大说明支撑脚在拖/滑。
  """
  in_contact, foot_speed = _contact_foot_speed(env, sensor_name, asset_cfg)
  return torch.sum(foot_speed * in_contact, dim=1) / torch.clamp(
    torch.sum(in_contact, dim=1), min=1.0
  )


def foot_slide_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  slip_threshold: float = 0.2,
) -> torch.Tensor:
  """接触脚中水平脚速超过 ``slip_threshold`` 的比例（0~1）。

  与离线检查脚本 ``slip_frac_gt_threshold`` 同阈值（0.2 m/s）同口径。
  """
  in_contact, foot_speed = _contact_foot_speed(env, sensor_name, asset_cfg)
  sliding = (foot_speed > slip_threshold).float() * in_contact
  return torch.sum(sliding, dim=1) / torch.clamp(torch.sum(in_contact, dim=1), min=1.0)


def joint_torque_ratio(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  effort_limit: float,
) -> torch.Tensor:
  """单关节 |扭矩| / 电机力矩规格。>1.0 = 该步扭矩超规格（电机跟不上）。"""
  torque = envs_mdp.joint_torque(env, asset_cfg)
  return torch.abs(torque) / effort_limit


def joint_velocity_ratio(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  velocity_limit: float,
) -> torch.Tensor:
  """单关节 |角速度| / 电机速度规格。>1.0 = 该步角速度超规格。"""
  vel = envs_mdp.joint_velocity(env, asset_cfg)
  return torch.abs(vel) / velocity_limit


def actuator_spec_violation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """该步是否有任一关节扭矩或角速度超电机规格（二值 0/1）。

  ``reduce="mean"`` 时聚合为"整段 episode 中超规格步的占比"，配合逐关节的
  ``joint_torque_ratio`` / ``joint_velocity_ratio`` 峰值，判断策略是否可部署。
  """
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]  # [B, N]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [B, N]
  torque_limits = torch.tensor(
    [_actuator_spec(name)[1] for name in PIKACHU_JOINT_NAMES],
    dtype=torch.float,
    device=torque.device,
  )
  velocity_limits = torch.tensor(
    [_actuator_spec(name)[0] for name in PIKACHU_JOINT_NAMES],
    dtype=torch.float,
    device=torque.device,
  )
  over_torque = torch.abs(torque) > torque_limits
  over_velocity = torch.abs(vel) > velocity_limits
  return (over_torque | over_velocity).any(dim=1).float()


def add_pikachu_telemetry_metrics(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add physical joint and base telemetry shared by every Pikachu task.

  The live Viser values remain signed. Episode summaries use the absolute peak,
  which is more useful when comparing actuator requirements across runs.
  """
  cfg.metrics["base_height_m"] = MetricsTermCfg(func=envs_mdp.root_height)

  # 脚底接地传感器：velocity 任务已定义，tracking 任务补一个（guard 防重名）。
  if not any(s.name == _FEET_CONTACT_SENSOR for s in cfg.scene.sensors or ()):
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
      ContactSensorCfg(
        name=_FEET_CONTACT_SENSOR,
        primary=ContactMatch(
          mode="subtree",
          pattern=r"^(left|right)_ankle_link$",
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      ),
    )

  foot_site_cfg = SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES)
  cfg.metrics["foot_slip_velocity_m_s"] = MetricsTermCfg(
    func=foot_slip_velocity,
    params={"asset_cfg": foot_site_cfg, "sensor_name": _FEET_CONTACT_SENSOR},
    reduce="mean",
  )
  cfg.metrics["foot_slide_fraction"] = MetricsTermCfg(
    func=foot_slide_fraction,
    params={"asset_cfg": foot_site_cfg, "sensor_name": _FEET_CONTACT_SENSOR},
    reduce="mean",
  )

  # 动作平滑度：action 二阶差分（a_t - 2 a_{t-1} + a_{t-2}）的均值，
  # 越小越平滑；高频抖动的策略此值会显著抬升，不能上实机。
  cfg.metrics["action_acc"] = MetricsTermCfg(
    func=envs_mdp.mean_action_acc,
    reduce="mean",
  )

  # 逐关节：位置/速度/扭矩 + 相对电机 spec 的峰值占用比。
  all_joint_cfg = SceneEntityCfg(
    "robot", joint_names=PIKACHU_JOINT_NAMES, preserve_order=True
  )
  for joint_name in PIKACHU_JOINT_NAMES:
    asset_cfg = SceneEntityCfg("robot", joint_names=(joint_name,))
    velocity_limit, effort_limit = _actuator_spec(joint_name)
    cfg.metrics[f"joint_position_rad/{joint_name}"] = MetricsTermCfg(
      func=envs_mdp.joint_position,
      params={"asset_cfg": asset_cfg},
      reduce="max_abs",
    )
    cfg.metrics[f"joint_velocity_rad_s/{joint_name}"] = MetricsTermCfg(
      func=envs_mdp.joint_velocity,
      params={"asset_cfg": asset_cfg},
      reduce="max_abs",
    )
    cfg.metrics[f"joint_torque_nm/{joint_name}"] = MetricsTermCfg(
      func=envs_mdp.joint_torque,
      params={"asset_cfg": asset_cfg},
      reduce="max_abs",
    )
    cfg.metrics[f"joint_torque_ratio/{joint_name}"] = MetricsTermCfg(
      func=joint_torque_ratio,
      params={"asset_cfg": asset_cfg, "effort_limit": effort_limit},
      reduce="max_abs",
    )
    cfg.metrics[f"joint_velocity_ratio/{joint_name}"] = MetricsTermCfg(
      func=joint_velocity_ratio,
      params={"asset_cfg": asset_cfg, "velocity_limit": velocity_limit},
      reduce="max_abs",
    )

  # 全关节超规格占比：任一关节扭矩或角速度超 spec 的步数占比（0~1）。
  cfg.metrics["actuator_spec_violation"] = MetricsTermCfg(
    func=actuator_spec_violation,
    params={"asset_cfg": all_joint_cfg},
    reduce="mean",
  )
