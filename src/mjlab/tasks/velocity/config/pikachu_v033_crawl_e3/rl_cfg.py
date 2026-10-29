"""Conservative E2 fine-tuning settings for the E3 crawl experiment."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e2.rl_cfg import crawl_e2_ppo_cfg


def crawl_e3_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = crawl_e2_ppo_cfg()
  cfg.max_iterations = 3000
  cfg.save_interval = 250
  cfg.algorithm.learning_rate = 5e-5
  cfg.algorithm.entropy_coef = 5e-4
  cfg.experiment_name = "pikachu_v033/crawl_e3"
  cfg.wandb_project = "pikachu-v033-velocity-crawl-e3"
  cfg.wandb_tags = (
    "v033",
    "crawl",
    "e3",
    "forward-refinement",
    "backward-retention",
  )
  return cfg
