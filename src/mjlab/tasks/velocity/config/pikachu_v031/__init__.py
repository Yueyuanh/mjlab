"""Pikachu v031（并联四杆膝，6.76 kg 整机）velocity task registration.

只注册 **一个** 任务：``Mjlab-Velocity-Flat-Pikachu-V031-T6``
（平地 + T6 奖励档，从零训 30k 轮）。

任务名里的 ``V031`` 后缀把它与 v03 / V025 的任务区分开；本包与
``config/pikachu_v03/`` **完全解耦** —— 资产、常量、奖励档全部独立，
只共用框架层（``velocity_env_cfg`` / ``mdp`` / ``pikachu_metrics``）。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pikachu_v031_b1_beta_bounded_env_cfg,
  pikachu_v031_e0_mirror_env_cfg,
  pikachu_v031_e1_clip_env_cfg,
  pikachu_v031_e2_symmetry_env_cfg,
  pikachu_v031_e3_yaw_huber_env_cfg,
  pikachu_v031_e4_curriculum_env_cfg,
  pikachu_v031_t6_flat_env_cfg,
  pikachu_v031_yaw_huber_flat_env_cfg,
)
from .rl_cfg import (
  pikachu_v031_b1_beta_bounded_ppo_runner_cfg,
  pikachu_v031_clip_ppo_runner_cfg,
  pikachu_v031_e0_mirror005_ppo_runner_cfg,
  pikachu_v031_e0_mirror010_ppo_runner_cfg,
  pikachu_v031_e1_clip_ppo_runner_cfg,
  pikachu_v031_e2_symmetry_ppo_runner_cfg,
  pikachu_v031_e3_yaw_huber_ppo_runner_cfg,
  pikachu_v031_e4_curriculum_ppo_runner_cfg,
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


_EXPERIMENTS = (
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-Clip",
    pikachu_v031_t6_flat_env_cfg,
    pikachu_v031_clip_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-Sym",
    pikachu_v031_t6_flat_env_cfg,
    pikachu_v031_sym_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-SymClip",
    pikachu_v031_t6_flat_env_cfg,
    pikachu_v031_sym_clip_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-YawHuber",
    pikachu_v031_yaw_huber_flat_env_cfg,
    pikachu_v031_yaw_huber_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-B1-BetaBounded",
    pikachu_v031_b1_beta_bounded_env_cfg,
    pikachu_v031_b1_beta_bounded_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005",
    pikachu_v031_e0_mirror_env_cfg,
    pikachu_v031_e0_mirror005_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror010",
    pikachu_v031_e0_mirror_env_cfg,
    pikachu_v031_e0_mirror010_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip",
    pikachu_v031_e1_clip_env_cfg,
    pikachu_v031_e1_clip_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry",
    pikachu_v031_e2_symmetry_env_cfg,
    pikachu_v031_e2_symmetry_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber",
    pikachu_v031_e3_yaw_huber_env_cfg,
    pikachu_v031_e3_yaw_huber_ppo_runner_cfg,
  ),
  (
    "Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum",
    pikachu_v031_e4_curriculum_env_cfg,
    pikachu_v031_e4_curriculum_ppo_runner_cfg,
  ),
)

for _task_id, _env_factory, _rl_factory in _EXPERIMENTS:
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=_env_factory(),
    play_env_cfg=_env_factory(play=True),
    rl_cfg=_rl_factory(),
    runner_cls=VelocityOnPolicyRunner,
  )
