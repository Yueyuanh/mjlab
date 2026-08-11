"""Pikachu velocity task registration.

注册两个任务（Rough / Flat），模仿 G1 的注册方式。任务 ID 采用
``Mjlab-Velocity-<地形>-Pikachu`` 命名，可用 ``uv run list-envs`` 查看。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pikachu_flat_env_cfg,
  pikachu_rough_env_cfg,
)
from .rl_cfg import pikachu_ppo_runner_cfg

# 粗糙地形 velocity 跟踪任务。
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pikachu",
  env_cfg=pikachu_rough_env_cfg(),
  play_env_cfg=pikachu_rough_env_cfg(play=True),
  rl_cfg=pikachu_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# 平坦地形 velocity 跟踪任务。
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu",
  env_cfg=pikachu_flat_env_cfg(),
  play_env_cfg=pikachu_flat_env_cfg(play=True),
  rl_cfg=pikachu_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
