"""RL configuration for the Pikachu walk-base (leg-only + arm-swing) tracking task.

纯腿 10 关节策略,从零重训(不复用 14 关节 walk-forward 的 checkpoint)。网络结构/
超参与 ``pikachu_walk`` 保持一致,仅 ``experiment_name`` 独立为 ``pikachu_walk_base``,
避免与 walk-forward 的 ``pikachu_tracking`` checkpoint 混用。训练中通过 ``arm_sine``
term 的摆臂课程逐步加入上肢扰动(见 ``envs/mdp/actions/arm_sine.py``)。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pikachu_walk_base_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration for Pikachu walk-base tracking."""
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
    experiment_name="pikachu_walk_base",
    wandb_project=wandb_project,
    wandb_tags=wandb_tags,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
