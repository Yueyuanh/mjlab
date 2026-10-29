"""V034 crawl task using the selected V033 E3 reward configuration."""

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity
from mjlab.tasks.velocity.config.pikachu_v034.env_cfgs import crawl_env_cfg
from mjlab.tasks.velocity.config.pikachu_v034.robot import SUPPORT_SITES
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import mdp


def v034_crawl_e3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = crawl_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 512
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.ranges.lin_vel_x = (-0.15, 0.20)
  command.ranges.lin_vel_y = (-0.04, 0.04)
  command.ranges.ang_vel_z = (-0.15, 0.15)
  if not play:
    command.rel_forward_envs = 0.40
    command.rel_backward_envs = 0.30
    command.rel_lateral_envs = 0.10
    command.rel_standing_envs = 0.10

  cfg.rewards["linear_velocity_huber"].weight = -0.75
  cfg.rewards["support_slip"].weight = -18.0
  cfg.rewards["gait_contacts"].weight = 0.0
  cfg.rewards["swing_air_time"] = RewardTermCfg(
    func=velocity.feet_air_time,
    weight=0.3,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.03,
      "threshold_max": 0.3,
      "command_name": "twist",
      "command_threshold": 0.04,
    },
  )
  cfg.rewards["elbow_window"] = RewardTermCfg(
    func=mdp.arm_posture_window,
    weight=-0.20,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("right_elbow_joint", "left_elbow_joint"),
        preserve_order=True,
      ),
      "tolerance_degrees": 8.0,
    },
  )
  cfg.rewards["compact_endpoints"] = RewardTermCfg(
    func=mdp.endpoint_distance_window,
    weight=-0.40,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        site_names=("right_hand", "right_foot", "left_hand", "left_foot"),
        preserve_order=True,
      ),
      "minimum": 0.10,
      "maximum": 0.25,
      "std": 0.08,
    },
  )
  feet = SceneEntityCfg(
    "robot", body_names=("right_ankle_link", "left_ankle_link"), preserve_order=True
  )
  for name, weight, endpoint in (
    ("stance_sole_normal", -0.80, False),
    ("stance_sole_endpoints", -0.20, True),
    ("forward_sole_normal", -0.45, False),
    ("forward_sole_endpoints", -0.10, True),
  ):
    cfg.rewards[name] = RewardTermCfg(
      func=mdp.directional_sole_cost
      if name.startswith("forward")
      else mdp.sole_support_cost,
      weight=weight,
      params={"asset_cfg": feet, "endpoint": endpoint},
    )
  cfg.rewards["stance_sole_outlier"] = RewardTermCfg(
    func=mdp.sole_support_cost,
    weight=-0.50,
    params={
      "asset_cfg": feet,
      "tolerance_degrees": 10.0,
      "scale_degrees": 5.0,
    },
  )
  cfg.rewards["ankle_limit_margin"] = RewardTermCfg(
    func=mdp.ankle_limit_margin,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("right_ankle_joint", "left_ankle_joint"),
        preserve_order=True,
      )
    },
  )
  cfg.rewards["gait_contacts_smooth_120"] = RewardTermCfg(
    func=mdp.smooth_gait_contacts,
    weight=1.50,
    params={"transition_seconds": 0.12},
  )
  cfg.rewards["forward_gait_mismatch_smooth_120"] = RewardTermCfg(
    func=mdp.directional_smooth_gait_mismatch,
    weight=-0.40,
    params={"transition_seconds": 0.12},
  )
  cfg.rewards["forward_velocity_shortfall"] = RewardTermCfg(
    func=mdp.forward_velocity_shortfall,
    weight=-0.25,
  )
  cfg.rewards["directional_progress"] = RewardTermCfg(
    func=mdp.directional_progress,
    weight=2.0,
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=common.action_acc_l2,
    weight=0.0,
  )
  cfg.rewards["swing_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=SUPPORT_SITES, preserve_order=True
  )
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = POLICY_ACTION_JOINT_NAMES
  action.preserve_order = True
  policy_joints = SceneEntityCfg(
    "robot", joint_names=POLICY_OBSERVATION_JOINT_NAMES, preserve_order=True
  )
  for group in cfg.observations.values():
    for name in ("joint_pos", "joint_vel"):
      group.terms[name].params["asset_cfg"] = policy_joints
  return cfg
