"""Low-noise warm-start tuning for compact four-support locomotion."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033_crawl_e1.rl_cfg import crawl_e1_ppo_cfg


def crawl_e2_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = crawl_e1_ppo_cfg()
  cfg.max_iterations = 5000
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.20,
    "std_type": "scalar",
    "std_range": (0.04, 0.25),
  }
  cfg.algorithm.entropy_coef = 0.001
  cfg.algorithm.learning_rate = 1e-4
  cfg.experiment_name = "pikachu_v033/crawl_e2"
  cfg.wandb_project = "pikachu-v033-velocity-crawl-e2"
  cfg.wandb_tags = ("v033", "crawl", "e2", "compact-support", "elbow-window")
  return cfg
