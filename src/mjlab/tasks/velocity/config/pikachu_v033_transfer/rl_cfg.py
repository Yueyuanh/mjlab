"""Small local PPO experiments, isolated from locomotion policies."""

from mjlab.tasks.velocity.config.pikachu_v033_crawl.rl_cfg import crawl_ppo_cfg


def transfer_ppo_cfg():
  cfg = crawl_ppo_cfg()
  cfg.actor.hidden_dims = (256, 128, 64)
  cfg.critic.hidden_dims = (256, 128, 64)
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.35,
    "std_type": "scalar",
    "std_range": (0.08, 0.5),
  }
  cfg.algorithm.learning_rate = 3e-4
  cfg.algorithm.entropy_coef = 0.003
  cfg.experiment_name = "pikachu_v033/tranfer"
  cfg.wandb_project = "pikachu-v033-transfer-no-reference"
  cfg.wandb_tags = ("v033", "biped-quadruped", "posture-command", "no-reference")
  cfg.max_iterations = 800
  cfg.save_interval = 400
  return cfg
