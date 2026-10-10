"""Pikachu「学步车」辅助 velocity 任务 RL 配置。

与基线 ``pikachu_ppo_runner_cfg`` 完全一致，仅改 ``experiment_name`` 区分训练日志，
wandb 按项目/标签由调用方传入（CLAUDE.md 规范）。
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu.rl_cfg import pikachu_ppo_runner_cfg


def pikachu_assisted_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create PPO runner config for the Pikachu「学步车」assisted velocity task."""
  cfg = pikachu_ppo_runner_cfg(wandb_project, wandb_tags)
  cfg.experiment_name = "pikachu_velocity_assisted"
  return cfg
