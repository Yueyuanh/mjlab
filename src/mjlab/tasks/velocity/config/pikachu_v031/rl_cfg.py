"""RL configuration for the Pikachu v031 velocity task.

PPO 超参与 v03 / 14dof V025 保持一致（同为小型人形，动作维度只差手臂），
只把 ``experiment_name`` 与 W&B 标识改成 v031 自己的，避免与 v03 的曲线混在
一起。``save_interval=500`` 是用户明确要求的存档节奏。

本模块提供 4 个 runner 档，用于隔离"单侧偏航 / hip_roll 不对称"的两个候选
根因（见 ``doc/pikachu_v031_velocity_20261010/``）：

- :func:`pikachu_v031_t6_ppo_runner_cfg` —— 原样基线（``clip_actions=null``、
  无对称约束）。
- :func:`pikachu_v031_clip_ppo_runner_cfg` —— 只加动作裁剪。
- :func:`pikachu_v031_sym_ppo_runner_cfg` —— 只加镜像数据增强 + mirror loss。
- :func:`pikachu_v031_sym_clip_ppo_runner_cfg` —— 两者都加（推荐部署档）。
- :func:`pikachu_v031_yaw_huber_ppo_runner_cfg` —— 奖励侧的单因素对照，环境换成
  :func:`pikachu_v031_yaw_huber_flat_env_cfg`（T6 + 偏航 Huber 项）。

**不要**在旧 checkpoint 上临时打开 ``clip_actions`` 后直接部署：旧策略依赖
越界目标产生的饱和扭矩，加裁剪会立刻摔倒，必须从头训练。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# 镜像增强函数路径（``rsl_rl.utils.resolve_callable`` 的 ``module:attr`` 形式）。
_MIRROR_FUNC = "mjlab.tasks.velocity.config.pikachu_v031.symmetry:mirror_augmentation"

# 动作裁剪上限。策略输出经此裁剪后再乘逐关节 scale，所以 hip_roll / hip_yaw
# 的处理后目标最多到 ±1.0 × 0.20 = ±0.20 rad，给 ±0.26 rad 机械硬限位留 0.06
# 余量，同时把"贴限位换扭矩"这条路封死。
CLIP_ACTIONS = 1.0


def _base_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    num_steps_per_env=24,
    # 从零训 30k 轮。
    max_iterations=30_000,
    # 每 500 轮存一次 checkpoint。
    save_interval=500,
  )


def pikachu_v031_t6_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Pikachu v031 flat velocity (T6)."""
  cfg = _base_runner_cfg()
  cfg.experiment_name = "pikachu_v031_velocity"
  # 每个注册任务必须有**独立**的 wandb_project / wandb_tags，否则多个任务的
  # 训练曲线会混在一起分不开。
  cfg.wandb_project = "pikachu-v031-velocity-flat"
  cfg.wandb_tags = ("velocity", "flat", "v031", "t6")
  return cfg


def pikachu_v031_clip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """T6 + 动作裁剪（隔离"无界动作目标"这一个因素）。"""
  cfg = _base_runner_cfg()
  # 注意：``clip_actions`` 是 **runner cfg** 的字段（``train.py`` 读
  # ``cfg.agent.clip_actions`` 传给 ``RslRlVecEnvWrapper``），不是 algorithm
  # 的字段。写成 ``cfg.algorithm.clip_actions`` 不会报错，但会静默失效。
  cfg.clip_actions = CLIP_ACTIONS
  cfg.experiment_name = "pikachu_v031_velocity_clip"
  cfg.wandb_project = "pikachu-v031-velocity-clip"
  cfg.wandb_tags = ("velocity", "flat", "v031", "t6", "clip")
  return cfg


def _symmetry_cfg(use_mirror_loss: bool, mirror_loss_coeff: float) -> dict:
  return {
    "data_augmentation_func": _MIRROR_FUNC,
    # 每个 minibatch 都追加一份镜像样本，actor/critic 的 loss 在两组上都算。
    "use_data_augmentation": True,
    # 额外的 MSE 项：惩罚策略在镜像观测上的输出与"原输出镜像"不一致。
    "use_mirror_loss": use_mirror_loss,
    "mirror_loss_coeff": mirror_loss_coeff,
  }


def pikachu_v031_sym_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """T6 + 镜像数据增强 + mirror loss（隔离"策略非镜像等变"这一个因素）。"""
  cfg = _base_runner_cfg()
  cfg.algorithm.symmetry_cfg = _symmetry_cfg(
    use_mirror_loss=True, mirror_loss_coeff=0.3
  )
  cfg.experiment_name = "pikachu_v031_velocity_sym"
  cfg.wandb_project = "pikachu-v031-velocity-sym"
  cfg.wandb_tags = ("velocity", "flat", "v031", "t6", "symmetry")
  return cfg


def pikachu_v031_yaw_huber_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """T6 环境 + 偏航 Huber 奖励项（隔离"偏航奖励饱和"这一个因素）。"""
  cfg = _base_runner_cfg()
  cfg.experiment_name = "pikachu_v031_velocity_yaw_huber"
  cfg.wandb_project = "pikachu-v031-velocity-yaw-huber"
  cfg.wandb_tags = ("velocity", "flat", "v031", "t6", "yaw-huber")
  return cfg


def pikachu_v031_sym_clip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """T6 + 动作裁剪 + 镜像约束（两者都加，候选部署档）。"""
  cfg = _base_runner_cfg()
  cfg.clip_actions = CLIP_ACTIONS
  cfg.algorithm.symmetry_cfg = _symmetry_cfg(
    use_mirror_loss=True, mirror_loss_coeff=0.3
  )
  cfg.experiment_name = "pikachu_v031_velocity_sym_clip"
  cfg.wandb_project = "pikachu-v031-velocity-sym-clip"
  cfg.wandb_tags = ("velocity", "flat", "v031", "t6", "symmetry", "clip")
  return cfg
