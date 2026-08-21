"""PPO configuration for the isolated Pikachu bounding task."""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..pikachu_quad.rl_cfg import pikachu_quad_ppo_runner_cfg


def pikachu_quad_bounding_ppo_runner_cfg(
  wandb_project: str,
  wandb_tags: tuple[str, ...],
) -> RslRlOnPolicyRunnerCfg:
  """Create a Quad-derived runner with an independent experiment directory."""
  return replace(
    pikachu_quad_ppo_runner_cfg(
      wandb_project=wandb_project,
      wandb_tags=wandb_tags,
    ),
    experiment_name="pikachu_velocity_quad_bounding",
  )
