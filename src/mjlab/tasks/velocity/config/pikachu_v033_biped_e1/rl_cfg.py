"""Upright E1 preserves the 18-action model and PPO mirror augmentation."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033.rl_cfg import (
  pikachu_v033_e0_ppo_runner_cfg,
)


def biped_e1_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pikachu_v033_e0_ppo_runner_cfg()
  cfg.experiment_name = "pikahcu_v033/biped_e1"
  cfg.wandb_project = "pikachu-v033-velocity-biped-e1"
  cfg.wandb_tags = (
    "v033",
    "biped",
    "e1",
    "arm-soft-window",
    "quiet-standing",
    "arm-leg-contact",
  )
  return cfg
