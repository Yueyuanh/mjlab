"""RL configuration for the Pikachu lite distillation (student) task.

网络结构与教师完全一致(512-256-128 ELU + Gaussian),这样教师权重能原样装进
``teacher`` 网络。学生 ``actor`` 的输入是 39 维可部署观测,教师是 65 维,
``critic`` 保持教师的 128 维特权观测。

算法换成 ``DaggerPPO``:``L = λ_PPO·L_PPO + λ_D·L_BC``,λ_D 从 1.0 退火到 0.1
(5000 迭代),BC 全程保留。
"""

from mjlab.rl import (
  RslRlDaggerPpoAlgorithmCfg,
  RslRlDaggerRunnerCfg,
  RslRlModelCfg,
)

_HIDDEN_DIMS = (512, 256, 128)


def _mlp(**kwargs) -> RslRlModelCfg:
  return RslRlModelCfg(hidden_dims=_HIDDEN_DIMS, activation="elu", **kwargs)


def pikachu_lite_student_runner_cfg(
  teacher_checkpoint: str,
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlDaggerRunnerCfg:
  """Create the DAgger + PPO distillation runner config.

  Args:
    teacher_checkpoint: 冻结教师的 ``model_*.pt`` 路径。
    wandb_project: W&B 项目名(与教师档区分)。
    wandb_tags: W&B 标签。
  """
  return RslRlDaggerRunnerCfg(
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher",),
    },
    actor=_mlp(
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=_mlp(obs_normalization=True),
    # 教师配置必须与 ``pikachu_lite`` 的 actor 逐项一致,否则 state_dict 装不上。
    teacher=_mlp(
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    teacher_checkpoint=teacher_checkpoint,
    algorithm=RslRlDaggerPpoAlgorithmCfg(
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
      # PHP Module 3:BC 从 1.0 退火到 0.1(不归零),前 5000 迭代退完。
      lambda_d_start=1.0,
      lambda_d_final=0.1,
      lambda_d_anneal_iters=5000,
      bc_loss_type="mse",
      bc_mask_terminations=True,
    ),
    experiment_name="pikachu_lite_distill",
    wandb_project=wandb_project,
    wandb_tags=wandb_tags,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
