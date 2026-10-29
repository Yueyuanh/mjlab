"""Sagittal left/right symmetry for the Pikachu V031 velocity policy."""

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

# Preserve the server's existing Sym/SymClip API and action layout. Mirror005
# continues to use pikachu_v031_data_augmentation below with its trained layout.
from .symmetry_legacy import (
  ACTUATOR_NAMES as ACTUATOR_NAMES,
)
from .symmetry_legacy import (
  JOINT_NAMES as JOINT_NAMES,
)
from .symmetry_legacy import (
  MIRROR_SIGN as MIRROR_SIGN,
)
from .symmetry_legacy import (
  _mirror_chunk as _mirror_chunk,
)
from .symmetry_legacy import (
  _mirror_foot_forces as _mirror_foot_forces,
)
from .symmetry_legacy import (
  _mirror_joints as _mirror_joints,
)
from .symmetry_legacy import (
  _partner as _partner,
)
from .symmetry_legacy import (
  _sign as _sign,
)
from .symmetry_legacy import (
  mirror_augmentation as mirror_augmentation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_VALIDATED_ENVS: WeakSet[object] = WeakSet()

# Actual MuJoCo order, asserted at runtime. The passive four-bar joints belong in
# joint observations even though they are absent from the action vector.
_JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "right_knee_up_joint",
  "right_knee_mid_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "left_knee_up_joint",
  "left_knee_mid_joint",
)
_ACTION_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_ankle_joint",
  "right_knee_up_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_ankle_joint",
  "left_knee_up_joint",
)

_JOINT_PERM = (7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6)
# V031 measured-axis mirror signs: hip pitch/roll/yaw, passive knee and ankle
# reverse; knee_up and knee_mid keep their sign.
_JOINT_SIGN = (-1, -1, -1, -1, -1, 1, 1) * 2
_ACTION_PERM = (5, 6, 7, 8, 9, 0, 1, 2, 3, 4)
_ACTION_SIGN = (-1, -1, -1, -1, 1) * 2

_ACTOR_LAYOUT = (
  ("base_lin_vel", 3),
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 14),
  ("joint_vel", 14),
  ("actions", 10),
  ("command", 3),
)
_CRITIC_LAYOUT = _ACTOR_LAYOUT + (
  ("foot_height", 2),
  ("foot_air_time", 2),
  ("foot_contact", 2),
  ("foot_contact_forces", 6),
)


def foot_contact_forces_heading_frame(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return per-foot net contact forces in the base-heading frame.

  ``ContactSensor(reduce="netforce")`` reports world-frame forces. A sagittal
  mirror cannot be applied correctly to those values after random reset yaw unless
  they are first rotated into the base-heading frame.
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


def _swap_signed(
  value: torch.Tensor, perm: tuple[int, ...], sign: tuple[int, ...]
) -> torch.Tensor:
  index = torch.tensor(perm, device=value.device)
  multiplier = torch.tensor(sign, device=value.device, dtype=value.dtype)
  return value.index_select(-1, index) * multiplier


def _mirror_group(value: torch.Tensor, group: str) -> torch.Tensor:
  layout = _ACTOR_LAYOUT if group == "actor" else _CRITIC_LAYOUT
  expected = sum(width for _, width in layout)
  if value.shape[-1] != expected:
    raise ValueError(
      f"V031 symmetry expected {group} width {expected}, got {value.shape[-1]}"
    )

  mirrored = value.clone()
  start = 0
  for name, width in layout:
    stop = start + width
    term = value[..., start:stop]
    if name in {"base_lin_vel", "projected_gravity"}:
      mirrored[..., start:stop] = term * term.new_tensor((1.0, -1.0, 1.0))
    elif name == "base_ang_vel":
      mirrored[..., start:stop] = term * term.new_tensor((-1.0, 1.0, -1.0))
    elif name in {"joint_pos", "joint_vel"}:
      mirrored[..., start:stop] = _swap_signed(term, _JOINT_PERM, _JOINT_SIGN)
    elif name == "actions":
      mirrored[..., start:stop] = _swap_signed(term, _ACTION_PERM, _ACTION_SIGN)
    elif name == "command":
      mirrored[..., start:stop] = term * term.new_tensor((1.0, -1.0, -1.0))
    elif name in {"foot_height", "foot_air_time", "foot_contact"}:
      mirrored[..., start:stop] = term.flip(-1)
    elif name == "foot_contact_forces":
      forces = term.reshape(*term.shape[:-1], 2, 3).flip(-2)
      forces = forces * forces.new_tensor((1.0, -1.0, 1.0))
      mirrored[..., start:stop] = forces.flatten(start_dim=-2)
    else:
      raise AssertionError(f"Unhandled V031 symmetry term: {name}")
    start = stop
  return mirrored


def _validate_env(env: RslRlVecEnvWrapper) -> None:
  raw = env.unwrapped
  if raw in _VALIDATED_ENVS:
    return
  robot = raw.scene["robot"]
  if tuple(robot.joint_names) != _JOINT_NAMES:
    raise ValueError(f"V031 joint order changed: {tuple(robot.joint_names)}")
  action_term = raw.action_manager.get_term("joint_pos")
  assert isinstance(action_term, JointPositionAction)
  action_names = tuple(action_term.target_names)
  if action_names != _ACTION_NAMES:
    raise ValueError(f"V031 action order changed: {action_names}")
  expected_terms = {
    "actor": [name for name, _ in _ACTOR_LAYOUT],
    "critic": [name for name, _ in _CRITIC_LAYOUT],
  }
  if raw.observation_manager.active_terms != expected_terms:
    raise ValueError(
      "V031 observation layout changed: "
      f"{raw.observation_manager.active_terms} != {expected_terms}"
    )
  _VALIDATED_ENVS.add(raw)


def pikachu_v031_data_augmentation(
  env: RslRlVecEnvWrapper,
  obs: TensorDictBase | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDictBase | None, torch.Tensor | None]:
  """Append one mirrored copy of every observation/action batch."""
  _validate_env(env)
  augmented_obs: TensorDictBase | None = None
  if obs is not None:
    mirrored_obs = obs.clone()
    for group in ("actor", "critic"):
      value = cast(torch.Tensor, obs[group])
      mirrored_obs[group] = _mirror_group(value, group)
    augmented_obs = td_cat((obs, mirrored_obs), dim=0)

  augmented_actions: torch.Tensor | None = None
  if actions is not None:
    mirrored_actions = _swap_signed(actions, _ACTION_PERM, _ACTION_SIGN)
    augmented_actions = torch.cat((actions, mirrored_actions), dim=0)
  return augmented_obs, augmented_actions
