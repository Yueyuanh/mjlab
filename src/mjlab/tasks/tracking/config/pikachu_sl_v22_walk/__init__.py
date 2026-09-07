"""Pikachu v2.2 sample-links forward-walking tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pikachu_sl_v22_walk_env_cfg
from .rl_cfg import pikachu_sl_v22_walk_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pikachu-Sl22-Walk",
  env_cfg=pikachu_sl_v22_walk_env_cfg(),
  play_env_cfg=pikachu_sl_v22_walk_env_cfg(play=True),
  rl_cfg=pikachu_sl_v22_walk_ppo_runner_cfg(
    wandb_project="pikachu-sl22-walk-forward",
    wandb_tags=("tracking", "walk", "forward", "0.163"),
  ),
  runner_cls=MotionTrackingOnPolicyRunner,
)
