"""PPO configuration for the V034 symmetric crawl task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v034_crawl_e3.rl_cfg import (
  v034_crawl_e3_ppo_cfg,
)


def v034_crawl_knee_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = v034_crawl_e3_ppo_cfg()
  # Keep the exploration range of the validated V033/V034 crawl configs.  The
  # biped pilot showed that raising the std cap and entropy above the proven
  # schedule pins the scalar std at the cap and makes training rollouts fall
  # from injected noise alone; see the biped experiment's
  # reports/exploration_noise_pilot.md.
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.25,
    "std_type": "scalar",
    "std_range": (0.04, 0.25),
  }
  cfg.algorithm.entropy_coef = 0.001
  cfg.max_iterations = 3000
  cfg.save_interval = 500
  cfg.experiment_name = "pikachu_v034/crawl_knee"
  cfg.wandb_project = "pikachu-v034-velocity-crawl-knee"
  cfg.wandb_tags = (
    "v034",
    "crawl",
    "symmetric-asset",
    "forward-progress",
    "native-action",
  )
  return cfg
