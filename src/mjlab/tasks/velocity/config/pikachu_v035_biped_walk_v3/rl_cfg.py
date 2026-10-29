"""V035 直立行走 v3 PPO：从 v1 末轮热启动，只加偏航 Huber 代价。

v1 的 ``std`` 收敛在 0.3 附近，这里把下界放到 0.25，给新加的奖励项
留出重新分配步态的空间，但不放开太大——步态骨架是现成的。
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk_v2.rl_cfg import (
  v035_biped_walk_v2_ppo_cfg,
)


def v035_biped_walk_v3_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v035_biped_walk_v2_ppo_cfg()
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.6,
    "std_type": "scalar",
    "std_range": (0.25, 0.8),
  }
  cfg.algorithm.entropy_coef = 0.006
  cfg.max_iterations = 1500
  cfg.save_interval = 250
  cfg.experiment_name = "pikachu_v035/biped_walk"
  cfg.wandb_project = "pikachu-v035-velocity-biped-walk"
  cfg.wandb_tags = ("v035", "biped", "walk", "yaw-huber")
  return cfg
