"""Pikachu 27-dof 四足全向 velocity 任务注册。

27-dof 拟真全身取四足构型(base 前俯、前后腿四点撑地)。策略驱动 16 走步关节
(腿 10 + 前腿 arm_pitch/roll/elbow 6),四足前方系命令/观测/奖励。任务 ID:
``Mjlab-Velocity-Flat-Pikachu-27dof-Quad``。独立命名空间,双足任务不受影响。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_27dof_quad_flat_env_cfg
from .rl_cfg import pikachu_27dof_quad_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-27dof-Quad",
  env_cfg=pikachu_27dof_quad_flat_env_cfg(),
  play_env_cfg=pikachu_27dof_quad_flat_env_cfg(play=True),
  rl_cfg=pikachu_27dof_quad_ppo_runner_cfg(
    wandb_project="pikachu-27dof-velocity-quad-flat",
    wandb_tags=("velocity", "27dof", "quad", "flat"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)
