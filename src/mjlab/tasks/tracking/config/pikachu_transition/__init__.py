"""Pikachu transition tracking task registration.

注册两个方向（Down / Up）的 Pikachu 状态转换 tracking 任务。任务 ID 采用
``Mjlab-Tracking-Flat-Pikachu-<方向>`` 命名，可用 ``uv run list-envs`` 查看。

参考动作数据来自 ``mjlab.mocap``（``mocap/npz/Pikachu-Transition-Down/Up.npz``），
已作为配置默认值打入 ``motion_file``，训练/评估无需再手动指定（play 例外，
见本目录 README）。
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  pikachu_transition_down_env_cfg,
  pikachu_transition_up_env_cfg,
)
from .rl_cfg import pikachu_transition_ppo_runner_cfg

# biped→quad（趴下）过渡。
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-Down",
  env_cfg=pikachu_transition_down_env_cfg(),
  play_env_cfg=pikachu_transition_down_env_cfg(play=True),
  rl_cfg=pikachu_transition_ppo_runner_cfg(
    wandb_project="pikachu-transition-down",
    wandb_tags=("transition", "down"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# quad→biped（站起）过渡。
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-Up",
  env_cfg=pikachu_transition_up_env_cfg(),
  play_env_cfg=pikachu_transition_up_env_cfg(play=True),
  rl_cfg=pikachu_transition_ppo_runner_cfg(
    wandb_project="pikachu-transition-up",
    wandb_tags=("transition", "up"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
