"""Register the isolated flat Pikachu quadruped bounding task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pikachu_quad_bounding_flat_env_cfg
from .rl_cfg import pikachu_quad_bounding_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pikachu-Quad-Bounding",
  env_cfg=pikachu_quad_bounding_flat_env_cfg(),
  play_env_cfg=pikachu_quad_bounding_flat_env_cfg(play=True),
  rl_cfg=pikachu_quad_bounding_ppo_runner_cfg(
    wandb_project="pikachu-velocity-quad-bounding",
    wandb_tags=("velocity", "quad", "bounding", "flat"),
  ),
  runner_cls=VelocityOnPolicyRunner,
)
