"""RL configuration for the Pikachu v031 velocity task.

PPO 超参与 v03 / 14dof V025 保持一致（同为小型人形，动作维度只差手臂），
只把 ``experiment_name`` 与 W&B 标识改成 v031 自己的，避免与 v03 的曲线混在
一起。``save_interval=500`` 是用户明确要求的存档节奏。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pikachu_v031_t6_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Pikachu v031 flat velocity (T6)."""
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
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pikachu_v031_velocity",
    # 每个注册任务必须有**独立**的 wandb_project / wandb_tags，否则多个任务的
    # 训练曲线会混在一起分不开。
    wandb_project="pikachu-v031-velocity-flat",
    wandb_tags=("velocity", "flat", "v031", "t6"),
    # 每 500 轮存一次 checkpoint。
    save_interval=500,
    num_steps_per_env=24,
    # 从零训 30k 轮。
    max_iterations=30_000,
  )
