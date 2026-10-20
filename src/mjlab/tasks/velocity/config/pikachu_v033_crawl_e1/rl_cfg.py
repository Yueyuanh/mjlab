"""Crawl E1 keeps the original PPO architecture and a separate log namespace."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl.rl_cfg import crawl_ppo_cfg


def crawl_e1_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = crawl_ppo_cfg()
  cfg.experiment_name = "pikahcu_v033/crawl_e1"
  cfg.wandb_project = "pikachu-v033-velocity-crawl-e1"
  cfg.wandb_tags = ("v033", "crawl", "e1", "flat", "full-action", "arm-soft-window")
  return cfg
