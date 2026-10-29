"""V035 直立行走 PPO：恢复 v031 E0 量级的高探索。"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_e1.rl_cfg import (
  v035_biped_e1_ppo_cfg,
)


def v035_biped_walk_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v035_biped_e1_ppo_cfg()
  # E1 的 init_std=0.25 / entropy=0.001 是给安静站立微调用的小探索，
  # 从零学走路会直接落进「站着不动」的局部最优。v031 E0 靠
  # init_std=1.0 / entropy=0.01 走出了 0.4 m/s，这里取中间档。
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.8,
    "std_type": "scalar",
    "std_range": (0.15, 1.0),
  }
  cfg.algorithm.entropy_coef = 0.005
  cfg.max_iterations = 5000
  cfg.save_interval = 500
  cfg.experiment_name = "pikachu_v035/biped_walk"
  cfg.wandb_project = "pikachu-v035-velocity-biped-walk"
  cfg.wandb_tags = ("v035", "biped", "walk", "direct-drive-knee", "v031-t6")
  return cfg
