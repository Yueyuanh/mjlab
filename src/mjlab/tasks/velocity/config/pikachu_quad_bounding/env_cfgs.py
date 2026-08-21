"""Flat Pikachu quadruped bounding task configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ..pikachu_quad.env_cfgs import pikachu_quad_flat_env_cfg
from .bounding_mdp import (
  bounding_contact_phases,
  bounding_flight_fraction,
  bounding_front_contact_fraction,
  bounding_pair_air_time,
  bounding_pair_desync_fraction,
  bounding_rear_contact_fraction,
  bounding_track_linear_velocity,
)

__all__ = ["pikachu_quad_bounding_flat_env_cfg"]


def pikachu_quad_bounding_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a forward-focused four-phase bounding task on flat terrain."""
  cfg = pikachu_quad_flat_env_cfg(play=play)

  # Bounding needs independent contact timing for the front and rear pairs.
  hands_ground_cfg = ContactSensorCfg(
    name="hands_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left|right)_arm_yaw_link$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hands_ground_cfg,)

  # Give only the critic privileged front-contact state. The actor interface is
  # unchanged from Quad, so this task does not alter deployment observations.
  critic = cfg.observations["critic"].terms
  critic["hand_air_time"] = ObservationTermCfg(
    func=mdp.foot_air_time,
    params={"sensor_name": hands_ground_cfg.name},
  )
  critic["hand_contact"] = ObservationTermCfg(
    func=mdp.foot_contact,
    params={"sensor_name": hands_ground_cfg.name},
  )
  critic["hand_contact_forces"] = ObservationTermCfg(
    func=mdp.foot_contact_forces,
    params={"sensor_name": hands_ground_cfg.name},
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if play:
    twist_cmd.ranges.lin_vel_x = (-0.4, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  else:
    # Start with forward bounding. Small lateral/yaw ranges retain robustness
    # without letting omnidirectional commands dominate gait discovery.
    twist_cmd.ranges.lin_vel_x = (0.15, 0.65)
    twist_cmd.ranges.lin_vel_y = (-0.05, 0.05)
    twist_cmd.ranges.ang_vel_z = (-0.1, 0.1)
    twist_cmd.rel_standing_envs = 0.1
    cfg.curriculum.pop("command_vel", None)

  # The parent Quad reward alternates the two rear feet and its linear tracker
  # penalizes vertical velocity. Both directly conflict with bounding.
  cfg.rewards.pop("air_time", None)
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=bounding_track_linear_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["upright"].weight = 0.75
  cfg.rewards["pose"].weight = 0.25
  cfg.rewards["angular_momentum"].weight = -0.005
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=envs_mdp.action_acc_l2,
    weight=-0.01,
  )
  cfg.rewards["foot_slip"].weight = -0.15
  cfg.rewards["soft_landing"].weight = -1e-4
  cfg.rewards["base_height"].weight = -0.02
  cfg.rewards["base_height"].params = {
    "target_height": 0.19,
    "std": 0.06,
  }

  cfg.rewards["bounding_phase"] = RewardTermCfg(
    func=bounding_contact_phases,
    weight=2.0,
    params={
      "hand_sensor_name": hands_ground_cfg.name,
      "foot_sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "target_stance_time": 0.12,
      "stance_std": 0.08,
      "target_flight_time": 0.06,
      "flight_std": 0.04,
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["bounding_pair_air_time"] = RewardTermCfg(
    func=bounding_pair_air_time,
    weight=3.0,
    params={
      "hand_sensor_name": hands_ground_cfg.name,
      "foot_sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "target_air_time": 0.2,
      "std": 0.07,
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["bounding_pair_desync"] = RewardTermCfg(
    func=bounding_pair_desync_fraction,
    weight=-1.0,
    params={
      "hand_sensor_name": hands_ground_cfg.name,
      "foot_sensor_name": "feet_ground_contact",
    },
  )

  cfg.metrics["bounding_front_contact_fraction"] = MetricsTermCfg(
    func=bounding_front_contact_fraction,
    params={"sensor_name": hands_ground_cfg.name},
  )
  cfg.metrics["bounding_rear_contact_fraction"] = MetricsTermCfg(
    func=bounding_rear_contact_fraction,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.metrics["bounding_flight_fraction"] = MetricsTermCfg(
    func=bounding_flight_fraction,
    params={
      "hand_sensor_name": hands_ground_cfg.name,
      "foot_sensor_name": "feet_ground_contact",
    },
  )
  cfg.metrics["bounding_pair_desync_fraction"] = MetricsTermCfg(
    func=bounding_pair_desync_fraction,
    params={
      "hand_sensor_name": hands_ground_cfg.name,
      "foot_sensor_name": "feet_ground_contact",
    },
  )

  return cfg
