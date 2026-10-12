"""Pikachu v031（并联四杆膝，6.76 kg 整机）velocity task registration.

注册 5 个任务，共用同一套 T6 奖励档，只隔离一个因素做对照：

- ``Mjlab-Velocity-Flat-Pikachu-V031-T6`` —— 原基线（无裁剪、无对称约束）。
- ``Mjlab-Velocity-Flat-Pikachu-V031-Clip`` —— 只加动作裁剪。
- ``Mjlab-Velocity-Flat-Pikachu-V031-Sym`` —— 只加镜像数据增强 + mirror loss。
- ``Mjlab-Velocity-Flat-Pikachu-V031-SymClip`` —— 两者都加。
- ``Mjlab-Velocity-Flat-Pikachu-V031-YawHuber`` —— 只加偏航 Huber 奖励项
  （环境换成 ``pikachu_v031_yaw_huber_flat_env_cfg``，其余同上）。

任务名里的 ``V031`` 后缀把它与 v03 / V025 的任务区分开；本包与
``config/pikachu_v03/`` **完全解耦** —— 资产、常量、奖励档全部独立，
只共用框架层（``velocity_env_cfg`` / ``mdp`` / ``pikachu_metrics``）。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pikachu_v031_t6_flat_env_cfg,
  pikachu_v031_yaw_huber_flat_env_cfg,
)
from .rl_cfg import (
  pikachu_v031_clip_ppo_runner_cfg,
  pikachu_v031_sym_clip_ppo_runner_cfg,
  pikachu_v031_sym_ppo_runner_cfg,
  pikachu_v031_t6_ppo_runner_cfg,
  pikachu_v031_yaw_huber_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V031-T6",
  env_cfg=pikachu_v031_t6_flat_env_cfg(),
  play_env_cfg=pikachu_v031_t6_flat_env_cfg(play=True),
  rl_cfg=pikachu_v031_t6_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V031-Clip",
  env_cfg=pikachu_v031_t6_flat_env_cfg(),
  play_env_cfg=pikachu_v031_t6_flat_env_cfg(play=True),
  rl_cfg=pikachu_v031_clip_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V031-Sym",
  env_cfg=pikachu_v031_t6_flat_env_cfg(),
  play_env_cfg=pikachu_v031_t6_flat_env_cfg(play=True),
  rl_cfg=pikachu_v031_sym_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V031-SymClip",
  env_cfg=pikachu_v031_t6_flat_env_cfg(),
  play_env_cfg=pikachu_v031_t6_flat_env_cfg(play=True),
  rl_cfg=pikachu_v031_sym_clip_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V031-YawHuber",
  env_cfg=pikachu_v031_yaw_huber_flat_env_cfg(),
  play_env_cfg=pikachu_v031_yaw_huber_flat_env_cfg(play=True),
  rl_cfg=pikachu_v031_yaw_huber_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
