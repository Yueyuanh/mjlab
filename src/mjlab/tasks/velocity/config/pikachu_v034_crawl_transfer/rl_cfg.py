"""Warm-start settings for V033-to-V034 crawl adaptation."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v034_crawl_e3.rl_cfg import (
  v034_crawl_e3_ppo_cfg,
)


def v034_crawl_transfer_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v034_crawl_e3_ppo_cfg()
  cfg.experiment_name = "pikachu_v034/crawl_transfer"
  cfg.wandb_project = "pikachu-v034-velocity-crawl-transfer"
  cfg.wandb_tags = (
    "v034",
    "crawl",
    "v033-actor-transfer",
    "crawl-pose-retarget",
  )
  return cfg
