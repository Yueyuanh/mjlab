"""V035 crawl PPO runner（v034 crawl-knee 的探索/学习率设定）。"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v034_crawl_knee.rl_cfg import (
  v034_crawl_knee_ppo_cfg,
)


def v035_crawl_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v034_crawl_knee_ppo_cfg()
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.25,
    "std_type": "scalar",
    "std_range": (0.04, 0.25),
  }
  cfg.algorithm.entropy_coef = 0.001
  cfg.max_iterations = 3000
  cfg.save_interval = 500
  cfg.experiment_name = "pikachu_v035/crawl"
  cfg.wandb_project = "pikachu-v035-velocity-crawl"
  cfg.wandb_tags = (
    "v035",
    "crawl",
    "direct-drive-knee",
    "forward-progress",
    "native-action",
  )
  return cfg
