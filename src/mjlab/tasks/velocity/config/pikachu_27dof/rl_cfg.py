"""RL configuration for Pikachu 27-dof sample-links velocity task.

策略网络/PPO 超参完全复用旧双足 velocity（``pikachu_ppo_runner_cfg``），只改
``experiment_name`` 为 ``pikachu_27dof_velocity``，让 checkpoint 与 W&B run
独立存放、不与旧任务混。
"""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..pikachu.rl_cfg import pikachu_ppo_runner_cfg


def pikachu_27dof_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner config for Pikachu 27-dof velocity task.

  Args:
    wandb_project: W&B project 名（如 ``pikachu-27dof-velocity-flat``）。
    wandb_tags: W&B run 标签，如 ``("velocity", "27dof", "flat")``。

  Returns:
    一个复用双足 PPO 超参、仅重命名 experiment 的配置实例。
  """
  return replace(
    pikachu_ppo_runner_cfg(wandb_project=wandb_project, wandb_tags=wandb_tags),
    experiment_name="pikachu_27dof_velocity",
  )