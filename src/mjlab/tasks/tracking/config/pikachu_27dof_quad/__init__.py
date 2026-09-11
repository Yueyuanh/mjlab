"""Pikachu 27-dof 四足前进 tracking 任务注册。

``Mjlab-Tracking-Flat-Pikachu-27dof-Quad-Walk``:27-dof 拟真全身取四足构型,策略驱动
16 走步关节(腿 10 + 前腿 arm_pitch/roll/elbow 6)跟踪四足参考;头/耳/尾固定。
参考运动当前为**占位**(见 ``pikachu_27dof_quad_constants.QUAD_27DOF_MOTION_FILE``),
任务框架/注册已就绪但不可训练,直到提供真实四足参考并回填。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pikachu_27dof_quad_walk_env_cfg
from .rl_cfg import pikachu_27dof_quad_walk_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-27dof-Quad-Walk",
  env_cfg=pikachu_27dof_quad_walk_env_cfg(),
  play_env_cfg=pikachu_27dof_quad_walk_env_cfg(play=True),
  rl_cfg=pikachu_27dof_quad_walk_ppo_runner_cfg(
    wandb_project="pikachu-27dof-quad-walk",
    wandb_tags=("tracking", "27dof", "quad"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
