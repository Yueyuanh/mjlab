"""Pikachu 27-dof hip-YRP forward-walk tracking task registration.

与 ``pikachu_27dof``（PRY）**完全解耦**：机器人换用 hip YRP 拆分序训练体，参考 npz
原样复用、运行时按名字对齐（``MotionCommandCfg.motion_body_column_names``），独立
checkpoint / W&B project，原 PRY 任务动不到。任务 ID：
``Mjlab-Tracking-Flat-Pikachu-27dof-Yrp-Walk``。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pikachu_27dof_yrp_walk_env_cfg
from .rl_cfg import pikachu_27dof_yrp_walk_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-27dof-Yrp-Walk",
  env_cfg=pikachu_27dof_yrp_walk_env_cfg(),
  play_env_cfg=pikachu_27dof_yrp_walk_env_cfg(play=True),
  rl_cfg=pikachu_27dof_yrp_walk_ppo_runner_cfg(
    wandb_project="pikachu-27dof-yrp-walk",
    wandb_tags=("tracking", "27dof", "yrp", "walk", "forward"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
