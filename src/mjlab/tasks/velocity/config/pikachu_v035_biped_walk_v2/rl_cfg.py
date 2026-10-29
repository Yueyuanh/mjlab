"""V035 直立行走 v2 PPO：从 v1 末轮热启动，只做偏航修正。

热启动意味着策略里已经有稳定的行走步态，这里需要的是「重新捏一下偏航」，
不是从零探索。v1 末轮 ``std`` 已经退到 0.15（下界），直接从 ckpt 续训的话
分布会锁死在 0.15，改不动行为；所以把 ``std_range`` 下界抬到 0.30，
分布每次采样都会被 clamp 回 0.30，等于给微调重新注入中等探索。
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk.rl_cfg import (
  v035_biped_walk_ppo_cfg,
)


def v035_biped_walk_v2_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v035_biped_walk_ppo_cfg()
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.8,
    "std_type": "scalar",
    "std_range": (0.30, 0.8),
  }
  cfg.algorithm.entropy_coef = 0.008
  cfg.max_iterations = 1500
  cfg.save_interval = 250
  cfg.experiment_name = "pikachu_v035/biped_walk"
  cfg.wandb_project = "pikachu-v035-velocity-biped-walk"
  cfg.wandb_tags = ("v035", "biped", "walk", "yaw-fix", "finetune")
  return cfg
