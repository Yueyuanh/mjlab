"""Reward compact same-side supports and small elbow excursions."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e1.env_cfgs import crawl_e1_env_cfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e1.mdp import arm_posture_window
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .mdp import ankle_limit_margin, endpoint_distance_window, sole_support_cost


def crawl_e2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = crawl_e1_env_cfg(play=play)
  cfg.scene.num_envs = 256 if not play else 1
  # E2 is a warm-start fine-tuning task: rehearse both directions immediately.
  # The inherited forward-only first 1000 iterations erased backward tracking.
  cfg.curriculum.pop("command_vel", None)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.15, 0.20)
  twist.ranges.lin_vel_y = (-0.04, 0.04)
  twist.ranges.ang_vel_z = (-0.15, 0.15)
  cfg.rewards["elbow_window"] = RewardTermCfg(
    func=arm_posture_window,
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
    func=endpoint_distance_window,
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
  cfg.rewards["stance_sole_normal"] = RewardTermCfg(
    func=sole_support_cost,
    weight=-0.8,
    params={"asset_cfg": feet},
  )
  cfg.rewards["stance_sole_endpoints"] = RewardTermCfg(
    func=sole_support_cost,
    weight=-0.20,
    params={"asset_cfg": feet, "endpoint": True},
  )
  cfg.rewards["stance_sole_outlier"] = RewardTermCfg(
    func=sole_support_cost,
    weight=-0.5,
    params={"asset_cfg": feet, "tolerance_degrees": 10.0, "scale_degrees": 5.0},
  )
  cfg.rewards["ankle_limit_margin"] = RewardTermCfg(
    func=ankle_limit_margin,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("right_ankle_joint", "left_ankle_joint"),
        preserve_order=True,
      )
    },
  )
  return cfg
