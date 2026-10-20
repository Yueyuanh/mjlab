"""Pikachu v033（并联四杆膝 + 双臂 8-dof）velocity task registration.

只注册 **一个** 任务：``Mjlab-Velocity-Flat-Pikachu-V033-E0``。

任务命名与 v031 的 ``Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005`` 对应：
环境侧同为 E0，runner 侧同为 mirror loss 0.05；按用户要求任务名只保留
``V033-E0``。

本包与 ``config/pikachu_v03/``、``config/pikachu_v031/`` **完全解耦** ——
资产、常量、奖励档、镜像函数全部独立，只共用框架层
（``velocity_env_cfg`` / ``mdp`` / ``pikachu_metrics``）。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_v033_e0_env_cfg
from .rl_cfg import pikachu_v033_e0_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V033-E0",
  env_cfg=pikachu_v033_e0_env_cfg(),
  play_env_cfg=pikachu_v033_e0_env_cfg(play=True),
  rl_cfg=pikachu_v033_e0_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
