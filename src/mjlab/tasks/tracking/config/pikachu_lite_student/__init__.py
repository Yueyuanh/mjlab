"""Pikachu lite **蒸馏档**:速度命令驱动 locomotion 的学生任务。

``Mjlab-Tracking-Flat-Pikachu-Lite-Distill`` = 教师档
(``Mjlab-Tracking-Flat-Pikachu-Lite``)的学生。教师用 65 维含参考的观测跟踪整库
25 条 rmg 参考;学生只用 44 维可部署观测(速度命令 + 本体感受 + 基座线速度),靠 DAgger + PPO
从冻结教师那里把"整库跟踪"的能力蒸馏过来。

部署形态:喂 ``(vx, vy, wz)``,命令项在参考库里最近邻匹配到对应 clip,策略输出
关节目标角 —— "发送速度控制命令自动匹配模型和数据,完成 locomotion"。

与教师档完全解耦:独立 config 包、独立 ``experiment_name`` /
``wandb_project``、独立 log 顶层目录。
"""

import os
from pathlib import Path

from mjlab.rl import DaggerOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import pikachu_lite_student_env_cfg
from .rl_cfg import pikachu_lite_student_runner_cfg

# 教师 checkpoint 的默认位置:本工作区 logs 下教师档最新的 model_*.pt。
# 只是便利默认值 —— 真正的权威来源是 ``--agent.teacher-checkpoint``。
# parents: [0]=本包 [1]=config [2]=tracking [3]=tasks [4]=mjlab(src) [5]=src
#          [6]=mjlab(repo) [7]=pikachu_playground
_TEACHER_LOG_ROOT = (
  Path(__file__).resolve().parents[7] / "pikachu_lite_locomotion" / "logs" / "rsl_rl"
)


def _default_teacher_checkpoint() -> str:
  """解析教师权重路径:环境变量优先,其次取教师档最新 checkpoint。"""
  env_path = os.environ.get("PIKACHU_TEACHER_CKPT", "")
  if env_path:
    return env_path
  try:
    # 通配 ``pikachu_lite_tracking*``:教师档换一版就是换一个顶层名
    # (``_signfix`` 是左腿符号修复后的那一版),写死 ``pikachu_lite_tracking``
    # 会静默挑到修复**之前**的教师 —— 蒸馏出来的学生会跟着学错的参考。
    # ``train_student.sh`` 里是同一处通配,两边必须一致。
    candidates = sorted(
      (
        p
        for d in _TEACHER_LOG_ROOT.glob("pikachu_lite_tracking*")
        for p in d.glob("*/model_*.pt")
      ),
      key=lambda p: (p.stat().st_mtime, p.name),
    )
  except OSError:
    return ""
  return str(candidates[-1]) if candidates else ""


register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-Lite-Distill",
  env_cfg=pikachu_lite_student_env_cfg(),
  play_env_cfg=pikachu_lite_student_env_cfg(play=True),
  rl_cfg=pikachu_lite_student_runner_cfg(
    teacher_checkpoint=_default_teacher_checkpoint(),
    wandb_project="pikachu-lite-tracking-distill",
    wandb_tags=("tracking", "lite", "rmgamp", "multiclip", "distill", "student"),
  ),
  runner_cls=DaggerOnPolicyRunner,
)
