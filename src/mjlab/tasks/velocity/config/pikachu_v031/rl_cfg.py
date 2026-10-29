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


def _e0_mirror_runner_cfg(
  *, mirror_loss_coeff: float, experiment_suffix: str
) -> RslRlOnPolicyRunnerCfg:
  """Create an E0 runner with only left/right symmetry learning enabled."""
  cfg = pikachu_v031_t6_ppo_runner_cfg()
  cfg.experiment_name = f"pikachu_v031_e0_mirror{experiment_suffix}"
  cfg.wandb_project = f"pikachu-v031-velocity-e0-mirror{experiment_suffix}"
  cfg.wandb_tags = (*cfg.wandb_tags, "e0-mirror", experiment_suffix)
  cfg.max_iterations = 50_000
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": (
      "mjlab.tasks.velocity.config.pikachu_v031.symmetry:pikachu_v031_data_augmentation"
    ),
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": mirror_loss_coeff,
  }
  return cfg


def pikachu_v031_e0_mirror005_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """E0 plus mirror loss 0.05; this is the primary experiment."""
  return _e0_mirror_runner_cfg(mirror_loss_coeff=0.05, experiment_suffix="005")


def pikachu_v031_e0_mirror010_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """E0 plus mirror loss 0.10; only run if the 0.05 screen is insufficient."""
  return _e0_mirror_runner_cfg(mirror_loss_coeff=0.10, experiment_suffix="010")


def pikachu_v031_b1_beta_bounded_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """E0 mirror005 with a naturally bounded Beta policy and safety clips."""
  cfg = _e0_mirror_runner_cfg(
    mirror_loss_coeff=0.05,
    experiment_suffix="005",
  )
  cfg.actor.distribution_cfg = {
    "class_name": "BetaDistribution",
    "action_range": (-1.0, 1.0),
  }
  # The Beta policy is already bounded. Keep this runner bound as a redundant
  # fail-safe for loading/export regressions, not as part of PPO's behavior.
  cfg.clip_actions = 1.0
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  cfg.experiment_name = "pikachu_v031_b1_beta_bounded"
  cfg.wandb_project = "pikachu-v031-velocity-b1-beta-bounded"
  cfg.wandb_tags = (
    "velocity",
    "flat",
    "v031",
    "b1-beta-bounded",
    "mirror005",
  )
  return cfg


def _improvement_runner_cfg(
  experiment_name: str, *, symmetry: bool
) -> RslRlOnPolicyRunnerCfg:
  cfg = pikachu_v031_t6_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.clip_actions = 1.0
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  cfg.wandb_tags = (*cfg.wandb_tags, "gait-symmetry-ablation")
  if symmetry:
    cfg.algorithm.symmetry_cfg = {
      "data_augmentation_func": (
        "mjlab.tasks.velocity.config.pikachu_v031.symmetry:"
        "pikachu_v031_data_augmentation"
      ),
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.30,
    }
  return cfg


def pikachu_v031_e1_clip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _improvement_runner_cfg("pikachu_v031_e1_clip", symmetry=False)


def pikachu_v031_e2_symmetry_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _improvement_runner_cfg("pikachu_v031_e2_symmetry", symmetry=True)


def pikachu_v031_e3_yaw_huber_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _improvement_runner_cfg("pikachu_v031_e3_yaw_huber", symmetry=True)


def pikachu_v031_e4_curriculum_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _improvement_runner_cfg("pikachu_v031_e4_curriculum", symmetry=True)


_MIRROR_FUNC = "mjlab.tasks.velocity.config.pikachu_v031.symmetry:mirror_augmentation"
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
