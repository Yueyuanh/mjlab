"""Crawl E1 changes rewards without restricting the full action interface."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl.env_cfgs import crawl_env_cfg

from .mdp import arm_posture_window

ARM_WINDOW_JOINTS = (
  "right_arm_roll_joint",
  "left_arm_roll_joint",
  "right_arm_yaw_joint",
  "left_arm_yaw_joint",
)


def crawl_e1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = crawl_env_cfg(play=play)
  cfg.rewards["arm_posture_window"] = RewardTermCfg(
    func=arm_posture_window,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=ARM_WINDOW_JOINTS, preserve_order=True
      ),
      "tolerance_degrees": 5.0,
    },
  )
  return cfg
