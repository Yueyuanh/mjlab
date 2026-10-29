"""Pikachu V034 的矢状面（``y = 0``）左右镜像，供 rsl_rl symmetry 扩展使用。

V034 沿用 18 个驱动关节 + 4 个被动四杆关节，但**关节符号表与 V033 不同**：

  - 观测里的 ``joint_pos`` / ``joint_vel`` 是 **22 列**（含 4 个被动四杆关节）；
  - 动作（以及 ``actions`` 观测）是 **18 列**（4 个被动关节没有 actuator）。

**本模块与 V033 完全解耦**：不 import ``pikachu_v033.symmetry`` 的任何东西，
关节顺序、符号表、观测 term 规则都在这里独立定义。

符号表由 FK 数值验证（逐关节摆 0.3 rad，闭链被动关节先求解，再比较左右链在
``y -> -y`` 反射下的位姿；残差 < 1e-6）。V034 与 V033 的关键差异：

  - ``knee_up`` **反号**（左腿负角才是屈膝）；
  - ``ankle``、``arm_yaw``、``elbow`` **同号**；
  - 被动 ``knee`` 反号、``knee_mid`` 同号。

不要凭"轴相同就同号"的直觉改表。

向量按极/轴矢量分类变换：

- 极矢量（线速度、重力）：``(x, y, z) -> (x, -y, z)``
- 轴矢量（角速度）：``(-x, y, -z)``
- 命令 ``(vx, vy, wz) -> (vx, -vy, -wz)``

顺序陷阱：``Entity.joint_names`` 与 ``Entity.actuator_names`` 的顺序不同
（V034 的动作顺序是「右腿 5 + 左腿 5 + 右臂 4 + 左臂 4」，手臂内部按
pitch / roll / yaw / elbow 排列）。本模块按**名字**构造置换与符号，
不硬编码下标；``_validate_env`` 在首次调用时断言实际顺序，模型改了会立刻报错。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from weakref import WeakSet

import torch
from tensordict import TensorDictBase
from tensordict import cat as td_cat

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_VALIDATED_ENVS: WeakSet[object] = WeakSet()

# MuJoCo 的关节顺序（= Entity.joint_names，也 = joint_pos/joint_vel 观测的列序）。
# 被动四杆关节 knee / knee_mid 也在里面，尽管它们没有 actuator。
_JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "right_knee_mid_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "left_knee_mid_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
)

# Entity.actuator_names 的顺序（= 动作向量的列序）。
_ACTION_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_up_joint",
  "right_ankle_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_up_joint",
  "left_ankle_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_joint",
)

# 左右交换时的关节角符号：+1 同号 / -1 反号。
# **FK 实测**（全零位下逐关节摆 0.3 rad，反射位姿残差 < 3e-6）。
MIRROR_SIGN = {
  "hip_pitch": -1.0,
  "hip_roll": -1.0,
  "hip_yaw": -1.0,
  "knee": -1.0,
  "knee_up": -1.0,
  "knee_mid": +1.0,
  "ankle": +1.0,
  "arm_pitch": -1.0,
  "arm_roll": -1.0,
  "arm_yaw": +1.0,
  "elbow": +1.0,
}

_ACTOR_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
)
_CRITIC_TERMS = _ACTOR_TERMS + (
  "foot_height",
  "foot_air_time",
  "foot_contact",
  "foot_contact_forces",
)

# 每个观测 term 的镜像规则。
_TERM_RULE = {
  "base_lin_vel": "polar",
  "base_ang_vel": "axial",
  "projected_gravity": "polar",
  "joint_pos": "joints",
  "joint_vel": "joints",
  "actions": "actuators",
  "command": "command",
  "foot_height": "swap",
  "foot_air_time": "swap",
  "foot_contact": "swap",
  "foot_contact_forces": "foot_forces",
}

_POLAR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)
_COMMAND = (1.0, -1.0, -1.0)


def _partner(name: str) -> str:
  # "left_" 是 5 字符、"right_" 是 6 字符，不能共用切片下标。
  prefix, rest = name.split("_", 1)
  if prefix == "left":
    return "right_" + rest
  if prefix == "right":
    return "left_" + rest
  raise ValueError(f"不是左右成对的名字: {name}")


def _joint_type(name: str) -> str:
  return name.split("_", 1)[1].removesuffix("_joint")


def _perm_and_sign(
  names: tuple[str, ...], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
  index = {n: i for i, n in enumerate(names)}
  perm = torch.tensor([index[_partner(n)] for n in names], device=device)
  sign = torch.tensor(
    [MIRROR_SIGN[_joint_type(n)] for n in names], device=device, dtype=dtype
  )
  return perm, sign


def _mirror_named(value: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
  if value.shape[-1] != len(names):
    raise ValueError(f"镜像维度应为 {len(names)}，实际 {value.shape[-1]}")
  perm, sign = _perm_and_sign(names, value.device, value.dtype)
  return value.index_select(-1, perm) * sign


def _mirror_axis(x: torch.Tensor, factors: tuple[float, ...]) -> torch.Tensor:
  if x.shape[-1] != 3:
    raise ValueError(f"三维向量观测应为 3 列，实际 {x.shape[-1]}")
  f = torch.tensor(factors, device=x.device, dtype=x.dtype)
  return x * f


def _mirror_swap(x: torch.Tensor, dim: int) -> torch.Tensor:
  """左右脚标量项（高度 / 滞空时间 / 接触标志）：交换两列即可。"""
  if dim % 2 != 0:
    raise ValueError(f"足部标量观测维度应为偶数，实际 {dim}")
  return torch.flip(x, dims=[-1])


def _mirror_foot_forces(x: torch.Tensor, dim: int) -> torch.Tensor:
  """接触力：先按脚交换，再对每只脚做 (x, -y, z) 的矢状面镜像。"""
  if dim % 3 != 0 or (dim // 3) % 2 != 0:
    raise ValueError(f"足部接触力维度应为 2 的倍数只脚 × 3 分量，实际 {dim}")
  n_feet = dim // 3
  v = x.reshape(*x.shape[:-1], n_feet, 3)
  v = torch.flip(v, dims=[-2])
  f = torch.tensor(_POLAR, device=x.device, dtype=x.dtype)
  return (v * f).reshape(*x.shape[:-1], dim)


def _mirror_chunk(rule: str, chunk: torch.Tensor, dim: int) -> torch.Tensor:
  if rule == "polar":
    return _mirror_axis(chunk, _POLAR)
  if rule == "axial":
    return _mirror_axis(chunk, _AXIAL)
  if rule == "command":
    return _mirror_axis(chunk, _COMMAND)
  if rule == "joints":
    return _mirror_named(chunk, _JOINT_NAMES)
  if rule == "actuators":
    return _mirror_named(chunk, _ACTION_NAMES)
  if rule == "swap":
    return _mirror_swap(chunk, dim)
  if rule == "foot_forces":
    return _mirror_foot_forces(chunk, dim)
  raise ValueError(f"未知镜像规则 {rule}")


def _mirror_obs(
  value: torch.Tensor,
  terms: tuple[str, ...] | list[str],
  dims: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
) -> torch.Tensor:
  """按 observation manager 的 term 名/维度切分并逐段镜像。"""
  total = sum(int(torch.tensor(d).prod()) for d in dims)
  if value.shape[-1] != total:
    raise ValueError(f"观测维度应为 {total}，实际 {value.shape[-1]}")
  pieces: list[torch.Tensor] = []
  offset = 0
  for name, d in zip(terms, dims, strict=True):
    n = int(torch.tensor(d).prod())
    chunk = value[..., offset : offset + n]
    offset += n
    rule = _TERM_RULE.get(name)
    if rule is None:
      raise KeyError(
        f"观测 term '{name}' 没有镜像规则；请在 symmetry.py 的 _TERM_RULE 里补上。"
      )
    pieces.append(_mirror_chunk(rule, chunk, n))
  return torch.cat(pieces, dim=-1)


def _mirror_action(actions: torch.Tensor) -> torch.Tensor:
  return _mirror_named(actions, _ACTION_NAMES)


def foot_contact_forces_heading_frame(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return per-foot net contact forces in the base-heading frame.

  ``ContactSensor(reduce="netforce")`` reports world-frame forces. A sagittal
  mirror cannot be applied correctly to those values after random reset yaw
  unless they are first rotated into the base-heading frame.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  forces_w = sensor.data.force
  asset: Entity = env.scene[asset_cfg.name]
  heading = yaw_quat(asset.data.root_link_pose_w[:, 3:7])
  heading = heading[:, None, :].expand(-1, forces_w.shape[1], -1)
  forces_b = quat_apply_inverse(heading, forces_w)
  transformed = torch.sign(forces_b) * torch.log1p(torch.abs(forces_b))
  return transformed.flatten(start_dim=1)


def _validate_env(env: RslRlVecEnvWrapper | ManagerBasedRlEnv) -> None:
  raw = env.unwrapped
  if raw in _VALIDATED_ENVS:
    return
  robot = raw.scene["robot"]
  if tuple(robot.joint_names) != _JOINT_NAMES:
    raise ValueError(f"V034 joint order changed: {tuple(robot.joint_names)}")
  action_term = raw.action_manager.get_term("joint_pos")
  assert isinstance(action_term, JointPositionAction)
  if tuple(action_term.target_names) != _ACTION_NAMES:
    raise ValueError(f"V034 action order changed: {tuple(action_term.target_names)}")
  expected_terms = {"actor": list(_ACTOR_TERMS), "critic": list(_CRITIC_TERMS)}
  if raw.observation_manager.active_terms != expected_terms:
    raise ValueError(
      "V034 observation layout changed: "
      f"{raw.observation_manager.active_terms} != {expected_terms}"
    )
  _VALIDATED_ENVS.add(raw)


def pikachu_v034_data_augmentation(
  env: RslRlVecEnvWrapper | ManagerBasedRlEnv,
  obs: TensorDictBase | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDictBase | None, torch.Tensor | None]:
  """Append one mirrored copy of every observation/action batch."""
  _validate_env(env)
  raw = env.unwrapped

  augmented_obs: TensorDictBase | None = None
  if obs is not None:
    mirrored_obs = obs.clone()
    for group in ("actor", "critic"):
      value = cast(torch.Tensor, obs[group])
      terms = raw.observation_manager.active_terms[group]
      dims = raw.observation_manager.group_obs_term_dim[group]
      mirrored_obs[group] = _mirror_obs(value, terms, dims)
    augmented_obs = td_cat((obs, mirrored_obs), dim=0)

  augmented_actions: torch.Tensor | None = None
  if actions is not None:
    augmented_actions = torch.cat((actions, _mirror_action(actions)), dim=0)
  return augmented_obs, augmented_actions
