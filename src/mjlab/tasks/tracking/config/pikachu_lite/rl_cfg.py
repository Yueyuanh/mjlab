"""RL configuration for the Pikachu lite multi-clip tracking task (teacher).

网络结构/超参与 ``pikachu_walk_base`` 保持一致(同一套 512-256-128 ELU + adaptive
KL PPO),只改三处与本任务语义相关的:

- ``experiment_name="pikachu_lite_tracking"`` —— 与 walk_base / 27dof 系列完全
  解耦,log 落到自己的顶层目录。
- ``wandb_project`` / ``wandb_tags`` 由调用方传入(``pikachu-lite-tracking``)。
- ``save_interval=500``、``max_iterations=10_000`` —— 需求指定的存盘粒度与规模。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pikachu_lite_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration for Pikachu lite tracking (teacher)."""
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
    experiment_name="pikachu_lite_tracking",
    wandb_project=wandb_project,
    wandb_tags=wandb_tags,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
