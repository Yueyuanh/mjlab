"""RL configuration for Pikachu quadruped velocity task.

四足与双足同为 14 dof 左右的小型类人形,PPO/网络超参复用双足
``pikachu_ppo_runner_cfg``(512→256→128,自适应 PPO),只改 experiment /
W&B 命名,保证两条任务曲线不混。训练不稳先调 ``learning_rate`` /
``desired_kl``。
"""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..pikachu.rl_cfg import pikachu_ppo_runner_cfg


def pikachu_quad_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner config for Pikachu quadruped velocity task.

  Args:
    wandb_project: W&B project 名(如 ``pikachu-velocity-quad-flat``)。
    wandb_tags: W&B run 标签,如 ``("velocity", "quad", "flat")``。

  Returns:
    一个 PPO 训练配置实例,网络/算法与双足 velocity 一致,
    仅重命名 experiment 以便 checkpoints 分开存放。
  """
  return replace(
    pikachu_ppo_runner_cfg(wandb_project=wandb_project, wandb_tags=wandb_tags),
    experiment_name="pikachu_velocity_quad",
  )
