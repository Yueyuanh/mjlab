"""Pikachu v033 velocity 任务的 RL 配置。

PPO 超参与 v031 保持一致（同为小型人形，动作维度只差手臂），但
``experiment_name``、W&B 项目与镜像函数都换成 v033 自己的，**不 import
v031 的 runner**，避免两代任务的曲线/函数引用混在一起。

本任务 ``Mjlab-Velocity-Flat-Pikachu-V033-E0`` 对应 v031 的
``E0-Mirror005`` 配方：

- 环境侧 E0：T6 + critic 接触力 heading 系修正（见 ``env_cfgs.py``）；
- runner 侧：镜像数据增强 + mirror loss = 0.05，训练 50k 轮，每 500 轮存档。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# 镜像增强函数必须用字符串引用，rsl_rl 在运行时按模块路径导入。
MIRROR_FUNC = (
  "mjlab.tasks.velocity.config.pikachu_v033.symmetry:pikachu_v033_data_augmentation"
)


def pikachu_v033_e0_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the V033 E0 runner (mirror augmentation + mirror loss 0.05)."""
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
      symmetry_cfg={
        "data_augmentation_func": MIRROR_FUNC,
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.05,
      },
    ),
    experiment_name="pikahcu_v033/biped_e0",
    # 每个注册任务必须有**独立**的 wandb_project / wandb_tags，否则多个任务
    # 的训练曲线会混在一起分不开。
    wandb_project="pikachu-v033-velocity-e0",
    wandb_tags=("velocity", "flat", "v033", "e0", "mirror005"),
    # 每 500 轮存一次 checkpoint。
    save_interval=500,
    num_steps_per_env=24,
    # E0 对照 v031 的 50k 轮设置。
    max_iterations=50_000,
  )
