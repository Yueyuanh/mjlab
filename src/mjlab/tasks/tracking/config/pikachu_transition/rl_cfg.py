"""RL configuration for Pikachu transition tracking task.

与 G1 tracking 使用相同的网络结构与 PPO 超参（同为小型人形、14 自由度，
见 ``config/g1/rl_cfg.py``）。实验名区分上下方向（由任务 ID 决定），
训练目录落在 ``logs/rsl_rl/pikachu_tracking/...``。

W&B 分离：``wandb_project`` / ``wandb_tags`` 为必填参数，每个任务（Down/Up）
必须传入独立的 project 与标签，避免不同任务的曲线混在同一个 project 里
（见 mjlab/CLAUDE.md「W&B 实验追踪」）。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pikachu_transition_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Pikachu transition tracking task.

  Args:
    wandb_project: W&B project 名，按任务分开（如 ``pikachu-transition-down``）。
    wandb_tags: W&B run 标签，如 ``("transition", "down")``。

  Returns:
    一个 PPO 训练配置实例，``experiment_name`` 统一为 ``pikachu_tracking``。
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
    experiment_name="pikachu_tracking",
    wandb_project=wandb_project,
    wandb_tags=wandb_tags,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
