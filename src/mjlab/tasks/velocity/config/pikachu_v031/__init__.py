"""Pikachu v031（并联四杆膝，6.76 kg 整机）velocity task registration.

只注册 **一个** 任务：``Mjlab-Velocity-Flat-Pikachu-V031-T6``
（平地 + T6 奖励档，从零训 30k 轮）。

任务名里的 ``V031`` 后缀把它与 v03 / V025 的任务区分开；本包与
``config/pikachu_v03/`` **完全解耦** —— 资产、常量、奖励档全部独立，
只共用框架层（``velocity_env_cfg`` / ``mdp`` / ``pikachu_metrics``）。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_v031_t6_flat_env_cfg
from .rl_cfg import pikachu_v031_t6_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V031-T6",
  env_cfg=pikachu_v031_t6_flat_env_cfg(),
  play_env_cfg=pikachu_v031_t6_flat_env_cfg(play=True),
  rl_cfg=pikachu_v031_t6_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
