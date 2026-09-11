"""RL configuration for Pikachu 27-dof quadruped velocity task.

四足同为一套 PPO 超参,网络/算法复用双足 ``pikachu_ppo_runner_cfg``,只改
``experiment_name`` 为 ``pikachu_27dof_velocity_quad`` 并独立 W&B 项目。
"""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..pikachu.rl_cfg import pikachu_ppo_runner_cfg


def pikachu_27dof_quad_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create PPO runner config for the 27-dof quadruped velocity task."""
  return replace(
    pikachu_ppo_runner_cfg(wandb_project=wandb_project, wandb_tags=wandb_tags),
    experiment_name="pikachu_27dof_velocity_quad",
  )
