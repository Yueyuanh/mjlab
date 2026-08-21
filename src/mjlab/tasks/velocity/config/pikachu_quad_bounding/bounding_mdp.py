"""Pikachu 四足 bounding 私有奖励与指标。

Bounding 使用四相循环：后肢组支撑推进 → 短腾空 → 前肢组支撑缓冲 →
短腾空。左右前肢必须同步，左右后肢也必须同步；这与原 Quad 任务的左右
交替步态不同，因此所有实现都放在独立命名空间中。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "bounding_contact_phases",
  "bounding_pair_air_time",
  "bounding_track_linear_velocity",
  "bounding_front_contact_fraction",
  "bounding_rear_contact_fraction",
  "bounding_flight_fraction",
  "bounding_pair_desync_fraction",
]

_LEFT_HAND = "left_arm_yaw_link"
_RIGHT_HAND = "right_arm_yaw_link"
_LEFT_FOOT = "left_ankle_link"
_RIGHT_FOOT = "right_ankle_link"
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bounding_track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track horizontal velocity without suppressing the bounding flight arc."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, UniformVelocityCommand)
  error = command.vel_command_w[:, :2] - asset.data.root_link_lin_vel_w[:, :2]
  return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def _command_active(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return total_command > command_threshold


def _pair_indices(
  sensor: ContactSensor,
  left_name: str,
  right_name: str,
) -> tuple[int, int]:
  return sensor.primary_names.index(left_name), sensor.primary_names.index(right_name)


def _pair_contact(
  sensor: ContactSensor,
  indices: tuple[int, int],
) -> torch.Tensor:
  assert sensor.data.found is not None
  return sensor.data.found[:, list(indices)] > 0


class bounding_contact_phases:
  """奖励按目标时长依次经过后支撑、腾空、前支撑、腾空四个 phase。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    hand_sensor: ContactSensor = env.scene[cfg.params["hand_sensor_name"]]
    foot_sensor: ContactSensor = env.scene[cfg.params["foot_sensor_name"]]
    self._hand_indices = _pair_indices(hand_sensor, _LEFT_HAND, _RIGHT_HAND)
    self._foot_indices = _pair_indices(foot_sensor, _LEFT_FOOT, _RIGHT_FOOT)
    # 0=rear stance, 1=flight after rear, 2=front stance,
    # 3=flight after front. Initial four-point support must first lift the hands.
    self._phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    hand_sensor_name: str,
    foot_sensor_name: str,
    command_name: str,
    target_stance_time: float,
    stance_std: float,
    target_flight_time: float,
    flight_std: float,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    """Return a dense timing score for valid bounding contact phases."""
    if stance_std <= 0.0 or flight_std <= 0.0:
      raise ValueError("bounding phase std values must be positive")

    hands: ContactSensor = env.scene[hand_sensor_name]
    feet: ContactSensor = env.scene[foot_sensor_name]
    hand_contact = _pair_contact(hands, self._hand_indices)
    foot_contact = _pair_contact(feet, self._foot_indices)

    assert hands.data.current_contact_time is not None
    assert hands.data.current_air_time is not None
    assert feet.data.current_contact_time is not None
    assert feet.data.current_air_time is not None

    front_stance = torch.all(hand_contact, dim=1) & ~torch.any(foot_contact, dim=1)
    rear_stance = torch.all(foot_contact, dim=1) & ~torch.any(hand_contact, dim=1)
    flight = ~torch.any(hand_contact, dim=1) & ~torch.any(foot_contact, dim=1)

    hand_contact_time = torch.amin(
      hands.data.current_contact_time[:, list(self._hand_indices)], dim=1
    )
    foot_contact_time = torch.amin(
      feet.data.current_contact_time[:, list(self._foot_indices)], dim=1
    )
    all_air_time = torch.cat(
      (
        hands.data.current_air_time[:, list(self._hand_indices)],
        feet.data.current_air_time[:, list(self._foot_indices)],
      ),
      dim=1,
    )
    flight_time = torch.amin(all_air_time, dim=1)

    # Only advance through the ordered four-phase cycle. Invalid or skipped
    # contacts receive no phase reward and cannot reset the expected phase.
    phase = self._phase
    phase = torch.where((phase == 0) & flight, 1, phase)
    phase = torch.where((phase == 1) & front_stance, 2, phase)
    phase = torch.where((phase == 2) & flight, 3, phase)
    phase = torch.where((phase == 3) & rear_stance, 0, phase)
    self._phase = phase

    front_score = (
      torch.exp(-torch.square((hand_contact_time - target_stance_time) / stance_std))
      * (phase == 2).float()
    )
    rear_score = (
      torch.exp(-torch.square((foot_contact_time - target_stance_time) / stance_std))
      * (phase == 0).float()
    )
    flight_score = (
      torch.exp(-torch.square((flight_time - target_flight_time) / flight_std))
      * ((phase == 1) | (phase == 3)).float()
    )
    score = (
      front_score * front_stance.float()
      + rear_score * rear_stance.float()
      + flight_score * flight.float()
    )
    active = _command_active(env, command_name, command_threshold)
    return score * active.float()

  def reset(self, env_ids: torch.Tensor) -> None:
    self._phase[env_ids] = 0


class bounding_pair_air_time:
  """奖励左右前肢或左右后肢同步落地，且整组腾空时间接近目标。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    hand_sensor: ContactSensor = env.scene[cfg.params["hand_sensor_name"]]
    foot_sensor: ContactSensor = env.scene[cfg.params["foot_sensor_name"]]
    self._hand_indices = _pair_indices(hand_sensor, _LEFT_HAND, _RIGHT_HAND)
    self._foot_indices = _pair_indices(foot_sensor, _LEFT_FOOT, _RIGHT_FOOT)
    # The initial rear support should launch into the first front-pair landing.
    self._next_pair = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    hand_sensor_name: str,
    foot_sensor_name: str,
    command_name: str,
    target_air_time: float,
    std: float,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    """Return landing rewards for synchronized front and rear limb pairs."""
    if std <= 0.0:
      raise ValueError(f"std must be positive, got {std}")

    hands: ContactSensor = env.scene[hand_sensor_name]
    feet: ContactSensor = env.scene[foot_sensor_name]
    assert hands.data.last_air_time is not None
    assert feet.data.last_air_time is not None

    hand_first_contact = hands.compute_first_contact(dt=env.step_dt)[
      :, list(self._hand_indices)
    ]
    foot_first_contact = feet.compute_first_contact(dt=env.step_dt)[
      :, list(self._foot_indices)
    ]
    hand_pair_landing = torch.all(hand_first_contact, dim=1)
    foot_pair_landing = torch.all(foot_first_contact, dim=1)
    hand_contact = _pair_contact(hands, self._hand_indices)
    foot_contact = _pair_contact(feet, self._foot_indices)
    valid_hand_landing = hand_pair_landing & ~torch.any(foot_contact, dim=1)
    valid_foot_landing = foot_pair_landing & ~torch.any(hand_contact, dim=1)

    hand_air_time = torch.mean(
      hands.data.last_air_time[:, list(self._hand_indices)], dim=1
    )
    foot_air_time = torch.mean(
      feet.data.last_air_time[:, list(self._foot_indices)], dim=1
    )
    hand_score = torch.exp(-torch.square((hand_air_time - target_air_time) / std))
    foot_score = torch.exp(-torch.square((foot_air_time - target_air_time) / std))
    reward_hand = valid_hand_landing & (self._next_pair == 0)
    reward_foot = valid_foot_landing & (self._next_pair == 1)
    reward = hand_score * reward_hand.float() + foot_score * reward_foot.float()
    self._next_pair = torch.where(reward_hand, 1, self._next_pair)
    self._next_pair = torch.where(reward_foot, 0, self._next_pair)
    active = _command_active(env, command_name, command_threshold)
    return reward * active.float()

  def reset(self, env_ids: torch.Tensor) -> None:
    self._next_pair[env_ids] = 0


def bounding_front_contact_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """当前两个前肢的接触比例。"""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.mean((sensor.data.found > 0).float(), dim=1)


def bounding_rear_contact_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """当前两个后肢的接触比例。"""
  return bounding_front_contact_fraction(env, sensor_name)


def bounding_flight_fraction(
  env: ManagerBasedRlEnv,
  hand_sensor_name: str,
  foot_sensor_name: str,
) -> torch.Tensor:
  """四肢同时腾空时返回 1，否则返回 0。"""
  hands: ContactSensor = env.scene[hand_sensor_name]
  feet: ContactSensor = env.scene[foot_sensor_name]
  assert hands.data.found is not None
  assert feet.data.found is not None
  any_contact = torch.any(hands.data.found > 0, dim=1) | torch.any(
    feet.data.found > 0, dim=1
  )
  return (~any_contact).float()


def bounding_pair_desync_fraction(
  env: ManagerBasedRlEnv,
  hand_sensor_name: str,
  foot_sensor_name: str,
) -> torch.Tensor:
  """前/后肢组内左右接触不同步的比例，0 最佳。"""
  hands: ContactSensor = env.scene[hand_sensor_name]
  feet: ContactSensor = env.scene[foot_sensor_name]
  assert hands.data.found is not None
  assert feet.data.found is not None
  hand_contact = hands.data.found > 0
  foot_contact = feet.data.found > 0
  hand_desync = hand_contact[:, 0] != hand_contact[:, 1]
  foot_desync = foot_contact[:, 0] != foot_contact[:, 1]
  return 0.5 * (hand_desync.float() + foot_desync.float())
