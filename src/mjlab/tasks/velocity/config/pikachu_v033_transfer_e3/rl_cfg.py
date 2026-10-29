"""Warm-start fine-tuning configuration for gentle E3 transitions."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033_transfer.rl_cfg import transfer_ppo_cfg


def transfer_e3_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = transfer_ppo_cfg()
  cfg.max_iterations = 3000
  cfg.save_interval = 250
  cfg.algorithm.learning_rate = 1e-4
  cfg.algorithm.entropy_coef = 0.001
  cfg.experiment_name = "pikachu_v033/transfer_e3"
  cfg.wandb_project = "pikachu-v033-transfer-e3-gentle"
  cfg.wandb_tags = (
    "v033",
    "transition",
    "e3",
    "gentle-up",
    "long-recovery",
    "no-motion-reference",
  )
  return cfg
