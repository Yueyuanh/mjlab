"""Pikachu 27-dof sample-links velocity task registration.

注册 27-dof 双足平地 velocity 任务。下肢 10 关节由策略行走，上肢（臂 8 +
耳 4 + 尾 2）作为随机摆动干扰（``RandomFlailAction``）。任务 ID：
``Mjlab-Velocity-Flat-Pikachu-27dof``。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_27dof_flat_env_cfg
from .rl_cfg import pikachu_27dof_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-27dof",
  env_cfg=pikachu_27dof_flat_env_cfg(),
  play_env_cfg=pikachu_27dof_flat_env_cfg(play=True),
  rl_cfg=pikachu_27dof_ppo_runner_cfg(
    wandb_project="pikachu-27dof-velocity-flat",
    wandb_tags=("velocity", "27dof", "flat"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)