"""RL configuration for Pikachu velocity task.

Pikachu 与 Unitree G1 使用相同的 PPO 超参即可（同为 14 自由度左右的小型
人形），网络结构与算法配置保持一致。若训练不稳定可先调 ``learning_rate``
或 ``desired_kl``，详见 root 目录 ``Architecture.md``。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pikachu_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Pikachu velocity task.

  Returns:
    一个 PPO 训练配置实例。网络为 actor/critic 双 MLP（512→256→128），
    高斯随机策略，自适应学习率 PPO。
  """
  return RslRlOnPolicyRunnerCfg(
    # 策略网络（actor）：输入观测 → 输出动作分布的均值。
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
    # 价值网络（critic）：输入观测 → 输出状态价值估计。
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
    # 实验名：决定 checkpoint 与 W&B run 的目录名。
    experiment_name="pikachu_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
