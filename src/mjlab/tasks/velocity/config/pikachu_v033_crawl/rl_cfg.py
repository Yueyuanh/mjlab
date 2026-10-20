"""PPO for crawl; independent tracking namespace, no biped symmetry assumptions."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def crawl_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      entropy_coef=0.005,
      clip_param=0.2,
      max_grad_norm=1,
    ),
    experiment_name="pikahcu_v033/crawl",
    wandb_project="pikachu-v033-velocity-crawl",
    wandb_tags=("v033", "crawl", "flat", "four-support"),
    num_steps_per_env=24,
    max_iterations=50_000,
    save_interval=500,
  )
