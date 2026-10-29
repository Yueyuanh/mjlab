"""V034 biped using the latest V033 E1 training semantics."""

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  POLICY_ACTION_JOINT_NAMES,
  POLICY_OBSERVATION_JOINT_NAMES,
  STAND_HEIGHT,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1.env_cfgs import (
  biped_e1_env_cfg,
)

from . import mdp
from .robot import biped_e1_robot_cfg


def v034_biped_e1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Keep E1 rewards while substituting V034 geometry and closed-chain timing."""
  cfg = biped_e1_env_cfg(play=play)
  cfg.scene.entities["robot"] = biped_e1_robot_cfg()
  cfg.scene.num_envs = 1 if play else 512
  cfg.sim.mujoco.timestep = 0.0025
  cfg.sim.mujoco.iterations = 20
  cfg.decimation = 8
  cfg.rewards["base_height"].params["target_height"] = STAND_HEIGHT
  cfg.rewards["directional_progress"] = RewardTermCfg(
    func=mdp.directional_progress,
    weight=2.0,
  )
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = POLICY_ACTION_JOINT_NAMES
  action.preserve_order = True
  policy_joints = SceneEntityCfg(
    "robot", joint_names=POLICY_OBSERVATION_JOINT_NAMES, preserve_order=True
  )
  for group in cfg.observations.values():
    for name in ("joint_pos", "joint_vel"):
      group.terms[name].params["asset_cfg"] = policy_joints
  return cfg
