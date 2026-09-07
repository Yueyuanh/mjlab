"""RL configuration for Pikachu v2.1 sample-links transition tracking task.

与旧 ``pikachu_transition`` 共用同一套网络结构与 PPO 超参（同构型复用），
仅实验名区分 superscript（``pikachu_sl21_transition``）。W&B 同样按方向
（Down/Up）分离 project 与标签。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pikachu_sl_v21_transition_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for v2.1 transition tracking task.

  Args:
    wandb_project: W&B project 名，按任务分开（如 ``pikachu-sl21-transition-down``）。
    wandb_tags: W&B run 标签，如 ``("transition", "down")``。

  Returns:
    一个 PPO 训练配置实例，``experiment_name`` 统一为 ``pikachu_sl21_transition``。
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pikachu_sl21_transition",
    wandb_project=wandb_project,
    wandb_tags=wandb_tags,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
