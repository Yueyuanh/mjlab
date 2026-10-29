"""V035 直立行走 v5 PPO：沿用 v1 的探索设置，只做 1500 轮微调。

不动 init_std / entropy，是为了让「角速度权重 2.0 → 6.0」成为唯一变量。
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk.rl_cfg import (
  v035_biped_walk_ppo_cfg,
)


def v035_biped_walk_v5_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v035_biped_walk_ppo_cfg()
  cfg.max_iterations = 1500
  cfg.save_interval = 250
  cfg.experiment_name = "pikachu_v035/biped_walk"
  cfg.wandb_project = "pikachu-v035-velocity-biped-walk"
  cfg.wandb_tags = ("v035", "biped", "walk", "ang-weight")
  return cfg
