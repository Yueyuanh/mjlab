"""V034 biped E1 runner derived from the latest V033 E1 configuration."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1.rl_cfg import (
  biped_e1_ppo_cfg,
)


def v034_biped_e1_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = biped_e1_ppo_cfg()
  # The V033 mirror mapping is not assumed valid for a newly imported body.
  # Keep the validated architecture and exploration settings, but train the
  # V034 policy without symmetry until a dedicated equivalence test exists.
  cfg.algorithm.symmetry_cfg = None
  cfg.max_iterations = 5000
  cfg.save_interval = 500
  cfg.experiment_name = "pikachu_v034/biped_e1"
  cfg.wandb_project = "pikachu-v034-velocity-biped-e1"
  cfg.wandb_tags = (
    "v034",
    "biped",
    "e1",
    "v033-framework",
    "actor-transfer",
  )
  return cfg
