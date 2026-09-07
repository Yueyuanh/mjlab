"""Pikachu 27-dof forward-walk tracking task registration.

``Mjlab-Tracking-Flat-Pikachu-27dof-Walk``：27-dof 全身拟真机器人做前进走步
tracking；策略只驱动腿部 10 关节跟踪纯腿参考运动，上肢 8 + 耳 4 + 尾 2 作为
随机摆动干扰（``RandomFlailAction``，训练用课程 ramp 摆幅）。策略观测覆盖全部
27 关节，能感知上肢扰动并主动平衡补偿。任务 ID：
``Mjlab-Tracking-Flat-Pikachu-27dof-Walk``。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pikachu_27dof_walk_env_cfg
from .rl_cfg import pikachu_27dof_walk_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-27dof-Walk",
  env_cfg=pikachu_27dof_walk_env_cfg(),
  play_env_cfg=pikachu_27dof_walk_env_cfg(play=True),
  rl_cfg=pikachu_27dof_walk_ppo_runner_cfg(
    wandb_project="pikachu-27dof-walk",
    wandb_tags=("tracking", "27dof", "walk", "forward"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)