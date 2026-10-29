"""V034 biped refinement for symmetric, knee-led locomotion."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_transfer.env_cfgs import (
  biped_transfer_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import mdp


def v034_biped_gait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Keep native V034 physics while correcting the measured gait failure modes."""
  cfg = biped_transfer_env_cfg(play=play, retarget_feet=True)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  if not play:
    # Retain a forward emphasis without the old 100% forward-only distribution.
    # Backward, lateral and heading samples prevent a one-sided local optimum.
    command.rel_forward_envs = 0.40
    command.rel_backward_envs = 0.20
    command.rel_lateral_envs = 0.15
    command.rel_standing_envs = 0.15
    command.rel_heading_envs = 1.0

  cfg.rewards["straight_base_roll_rate"] = RewardTermCfg(
    func=mdp.straight_base_roll_rate,
    weight=-0.08,
    params={"deadzone": 0.45, "std": 0.75},
  )
  cfg.rewards["swing_knee_separation"] = RewardTermCfg(
    func=mdp.swing_knee_separation,
    weight=-0.05,
    params={
      "sensor_name": "feet_ground_contact",
      "margin": 0.07,
      "std": 0.07,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("right_knee_up_joint", "left_knee_up_joint"),
        preserve_order=True,
      ),
    },
  )
  return cfg
