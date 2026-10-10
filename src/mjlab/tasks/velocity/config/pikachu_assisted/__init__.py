"""Pikachu velocity「学步车」辅助任务注册。

衍生自 velocity Pikachu Flat（14dof），给 base 点加**硬虚拟碰撞护栏**（上下两个
水平平面，纯碰撞不施力）+ upright 奖励增强。``tasks/__init__.py`` 的
``import_packages`` 会递归发现本目录并触发注册。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_assisted_flat_env_cfg
from .rl_cfg import pikachu_assisted_ppo_runner_cfg

# 平坦地形「学步车」辅助 velocity 任务。
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-Assisted",
  env_cfg=pikachu_assisted_flat_env_cfg(),
  play_env_cfg=pikachu_assisted_flat_env_cfg(play=True),
  rl_cfg=pikachu_assisted_ppo_runner_cfg(
    wandb_project="pikachu-velocity-flat-assisted",
    wandb_tags=("velocity", "flat", "assisted"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)
