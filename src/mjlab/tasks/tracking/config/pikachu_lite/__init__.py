"""Pikachu lite multi-clip tracking task registration (teacher).

``Mjlab-Tracking-Flat-Pikachu-Lite`` = ``pikachu_constants_lite`` 碰撞体 +
``mocap/npz/pikachu_rmgamp_npz`` 全部 25 条 rmg 速度档参考。一个策略同时跟踪
整库(``MultiMotionCommand``),是后续"速度命令 -> 自动匹配参考"蒸馏的教师。

与既有任务完全解耦:独立 config 包、独立 ``experiment_name`` /
``wandb_project``、独立 log 顶层目录。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pikachu_lite_env_cfg
from .rl_cfg import pikachu_lite_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-Lite",
  env_cfg=pikachu_lite_env_cfg(),
  play_env_cfg=pikachu_lite_env_cfg(play=True),
  rl_cfg=pikachu_lite_ppo_runner_cfg(
    wandb_project="pikachu-lite-tracking",
    wandb_tags=("tracking", "lite", "rmgamp", "multiclip", "teacher"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
