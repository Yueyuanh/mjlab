"""RL configuration for Pikachu v03 velocity task.

PPO 超参与 14dof V025 保持一致（同为小型人形，动作维度只差 4 个手臂关节）。
``save_interval=500`` 是用户明确要求的存档节奏（约每 500 轮一个 ``.pt``）。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pikachu_v03_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the Pikachu v03 velocity task."""
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
    experiment_name="pikachu_v03_velocity",
    wandb_project=wandb_project,
    wandb_tags=wandb_tags,
    # 每 500 轮存一次 checkpoint（用户要求）。
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def pikachu_v03_t6_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Conservative fine-tuning runner for the isolated T6 reward profile."""
  cfg = pikachu_v03_ppo_runner_cfg(
    wandb_project="pikachu-v03-velocity-flat",
    wandb_tags=("velocity", "flat", "v03", "t6", "gait-refine"),
  )
  # Keep the same experiment root so --agent.load-run can resolve the r06
  # checkpoint; the separate task ID and mandatory t6 run name distinguish it.
  cfg.experiment_name = "pikachu_v03_velocity"
  # This is an additional iteration budget after loading r06-best, not a
  # global target iteration number (Mjlab resumes from the checkpoint's iter).
  cfg.max_iterations = 12_000
  return cfg
