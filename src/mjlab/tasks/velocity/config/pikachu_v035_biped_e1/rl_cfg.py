"""V035 直立 runner：沿用已验证的 v034 PPO 设定，无对称增强。"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_e1.rl_cfg import (
  v034_biped_e1_ppo_cfg,
)


def v035_biped_e1_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v034_biped_e1_ppo_cfg()
  cfg.algorithm.symmetry_cfg = None
  cfg.max_iterations = 5000
  cfg.save_interval = 500
  cfg.experiment_name = "pikachu_v035/biped_e1"
  cfg.wandb_project = "pikachu-v035-velocity-biped-e1"
  cfg.wandb_tags = ("v035", "biped", "e1", "direct-drive-knee", "v033-framework")
  return cfg
