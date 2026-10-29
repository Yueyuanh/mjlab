"""E3 keeps E2 backward locomotion while focusing updates on forward gait."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e2.env_cfgs import crawl_e2_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import mdp


def crawl_e3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = crawl_e2_env_cfg(play=play)
  cfg.scene.num_envs = 512 if not play else 1
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.15, 0.20)
  twist.ranges.lin_vel_y = (-0.04, 0.04)
  twist.ranges.ang_vel_z = (-0.15, 0.15)
  if not play:
    # Explicit buckets prevent the already-good backward gait from being
    # forgotten while allocating more samples to the weaker forward gait.
    twist.rel_forward_envs = 0.40
    twist.rel_backward_envs = 0.30
    twist.rel_lateral_envs = 0.10
    twist.rel_standing_envs = 0.10

  feet = SceneEntityCfg(
    "robot", body_names=("right_ankle_link", "left_ankle_link"), preserve_order=True
  )
  cfg.rewards["forward_sole_normal"] = RewardTermCfg(
    func=mdp.directional_sole_cost,
    weight=-0.45,
    params={"asset_cfg": feet},
  )
  cfg.rewards["forward_sole_endpoints"] = RewardTermCfg(
    func=mdp.directional_sole_cost,
    weight=-0.10,
    params={"asset_cfg": feet, "endpoint": True},
  )
  cfg.rewards["forward_gait_mismatch"] = RewardTermCfg(
    func=mdp.directional_gait_mismatch,
    weight=0.0,
  )
  for milliseconds in (80, 120):
    cfg.rewards[f"gait_contacts_smooth_{milliseconds:03d}"] = RewardTermCfg(
      func=mdp.smooth_gait_contacts,
      weight=1.50 if milliseconds == 120 else 0.0,
      params={"transition_seconds": milliseconds / 1000},
    )
    cfg.rewards[f"forward_gait_mismatch_smooth_{milliseconds:03d}"] = RewardTermCfg(
      func=mdp.directional_smooth_gait_mismatch,
      weight=-0.40 if milliseconds == 120 else 0.0,
      params={"transition_seconds": milliseconds / 1000},
    )
  cfg.rewards["gait_contacts"].weight = 0.0
  cfg.rewards["forward_velocity_shortfall"] = RewardTermCfg(
    func=mdp.forward_velocity_shortfall,
    weight=-0.25,
  )
  # Keep the historical E3 behavior at zero while allowing audited E3 sweeps
  # to activate a second-difference penalty without registering throwaway tasks.
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=common.action_acc_l2,
    weight=0.0,
  )
  return cfg
