"""V035 直立行走 v4 PPO：膝关节换成 DC 电机模型后的 PPO 配置。

v2 末轮 ``std`` 在 0.57 附近；这里把下界放到 0.25，
给「改换膝关节动力学」留出足够探索，但不要太大——步态骨架是现成的，
需要的是重新分配摆动相的时间，不是从零探索。
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk_v2.rl_cfg import (
  v035_biped_walk_v2_ppo_cfg,
)


def v035_biped_walk_v4_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v035_biped_walk_v2_ppo_cfg()
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.6,
    "std_type": "scalar",
    "std_range": (0.25, 0.8),
  }
  cfg.algorithm.entropy_coef = 0.006
  cfg.max_iterations = 2000
  cfg.save_interval = 250
  cfg.experiment_name = "pikachu_v035/biped_walk"
  cfg.wandb_project = "pikachu-v035-velocity-biped-walk"
  cfg.wandb_tags = ("v035", "biped", "walk", "dc-knee", "sim2real")
  return cfg
