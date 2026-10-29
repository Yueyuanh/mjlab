"""Local PPO baselines; mirror augmentation requires a new V034 validation."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def ppo_runner_cfg(mode="biped"):
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True
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
    experiment_name=f"pikachu_v034/{mode}",
    logger="tensorboard",
    wandb_project=f"pikachu-v034-{mode}",
    wandb_tags=("v034", mode, "baseline"),
    num_steps_per_env=24,
    max_iterations=5000,
    save_interval=250,
    clip_actions=3.0,
  )
