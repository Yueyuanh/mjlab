"""Pikachu v03（并联四杆膝）velocity task registration.

任务 ID：``Mjlab-Velocity-{Flat,Rough}-Pikachu-V03`` 与
``Mjlab-Velocity-Flat-Pikachu-V03-T6``，可用 ``uv run list-envs`` 查看。

⚠️ 任务名里带 ``V03`` 后缀，与 14dof V025 的 ``Mjlab-Velocity-Flat-Pikachu``
区分开；两者共用同一套 PPO 超参但资产、动作维度（10 vs 14）完全不同。

``-T6`` 是 R06 基线上的保守微调档（改奖励 + 命令覆盖，不动 actor/critic 形状），
用 weight 6.3 kg 模型与降额后的 J3507 力矩上限，见 ``env_cfgs.py``。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pikachu_v03_flat_env_cfg,
  pikachu_v03_rough_env_cfg,
  pikachu_v03_t6_flat_env_cfg,
)
from .rl_cfg import pikachu_v03_ppo_runner_cfg, pikachu_v03_t6_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pikachu-V03",
  env_cfg=pikachu_v03_rough_env_cfg(),
  play_env_cfg=pikachu_v03_rough_env_cfg(play=True),
  rl_cfg=pikachu_v03_ppo_runner_cfg(
    wandb_project="pikachu-v03-velocity-rough",
    wandb_tags=("velocity", "rough", "v03"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V03",
  env_cfg=pikachu_v03_flat_env_cfg(),
  play_env_cfg=pikachu_v03_flat_env_cfg(play=True),
  rl_cfg=pikachu_v03_ppo_runner_cfg(
    wandb_project="pikachu-v03-velocity-flat",
    wandb_tags=("velocity", "flat", "v03"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-V03-T6",
  env_cfg=pikachu_v03_t6_flat_env_cfg(),
  play_env_cfg=pikachu_v03_t6_flat_env_cfg(play=True),
  rl_cfg=pikachu_v03_t6_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
