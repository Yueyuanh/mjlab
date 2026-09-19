"""Pikachu 27-dof hip-YRP sample-links velocity task registration.

与 ``pikachu_27dof``（PRY）**完全解耦**：机器人换用 hip YRP 拆分序的训练体
``pikachu_27dof_train_yrp.xml``，独立 checkpoint / W&B project，原 PRY 任务动不到。
任务 ID：``Mjlab-Velocity-Flat-Pikachu-27dof-Yrp``。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_27dof_yrp_flat_env_cfg
from .rl_cfg import pikachu_27dof_yrp_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-27dof-Yrp",
  env_cfg=pikachu_27dof_yrp_flat_env_cfg(),
  play_env_cfg=pikachu_27dof_yrp_flat_env_cfg(play=True),
  rl_cfg=pikachu_27dof_yrp_ppo_runner_cfg(
    wandb_project="pikachu-27dof-yrp-velocity-flat",
    wandb_tags=("velocity", "27dof", "yrp", "flat"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)
