"""Pikachu walk-base tracking task registration.

``Mjlab-Tracking-Flat-Pikachu-Walk-Base`` 是手臂摆动力学**真消融**实验:策略输入/输出/
参考全部只含 10 个腿部关节,手臂从零维副作用 ``ArmSineAction`` 独立施加正弦扰动,
训练中用摆臂课程从 0 ramp 到上限。该任务从零重训 10 关节纯腿策略(``experiment_name
= pikachu_walk_base``),不复用 14 关节 walk-forward 模型。play 用 ``ArmSwingViewer``
滑条实时控制摆幅/频率。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pikachu_walk_base_env_cfg
from .rl_cfg import pikachu_walk_base_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-Walk-Base",
  env_cfg=pikachu_walk_base_env_cfg(),
  play_env_cfg=pikachu_walk_base_env_cfg(play=True),
  rl_cfg=pikachu_walk_base_ppo_runner_cfg(
    wandb_project="pikachu-walk-base",
    wandb_tags=("tracking", "walk", "base", "ablation"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
