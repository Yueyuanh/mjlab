"""Pikachu v2.2 sample-links 四足全向 velocity 任务注册。

独立命名空间,只加 flat 起步任务。任务 ID:
``Mjlab-Velocity-Flat-Pikachu-Sl22-Quad``。共享双足/旧 quad 任务完全不受影响。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_sl_v22_quad_flat_env_cfg
from .rl_cfg import pikachu_sl_v22_quad_ppo_runner_cfg

# 平坦地形四足全向 velocity 任务。
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-Sl22-Quad",
  env_cfg=pikachu_sl_v22_quad_flat_env_cfg(),
  play_env_cfg=pikachu_sl_v22_quad_flat_env_cfg(play=True),
  rl_cfg=pikachu_sl_v22_quad_ppo_runner_cfg(
    wandb_project="pikachu-sl22-velocity-quad-flat",
    wandb_tags=("velocity", "quad", "flat", "0.04"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)
