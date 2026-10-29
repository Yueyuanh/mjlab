"""V034 walking transfer with V033 policy coordinates and native physics."""

import copy
from dataclasses import fields

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.pikachu_v034_biped_e1.env_cfgs import (
  v034_biped_e1_env_cfg,
)

from . import mdp


def biped_transfer_env_cfg(
  play: bool = False, retarget_feet: bool = False, gain_scale: float = 1.0
) -> ManagerBasedRlEnvCfg:
  cfg = v034_biped_e1_env_cfg(play=play)
  original = cfg.actions["joint_pos"]
  assert isinstance(original, JointPositionActionCfg)
  cfg.actions["joint_pos"] = mdp.TransferJointPositionActionCfg(
    retarget_feet=retarget_feet,
    **{field.name: getattr(original, field.name) for field in fields(original)},
  )
  for group in cfg.observations.values():
    group.terms["joint_pos"].func = mdp.joint_pos_reference
    group.terms["joint_vel"].func = mdp.joint_vel_reference
  # Elbow native axes are reversed. Retain the physical V033 arm window.
  bounds = list(cfg.rewards["arm_position_window"].params["bounds_degrees"])
  for index in (3, 7):
    lower, upper = bounds[index]
    bounds[index] = (-upper, -lower)
  cfg.rewards["arm_position_window"].params["bounds_degrees"] = tuple(bounds)
  if gain_scale != 1.0:
    robot = cfg.scene.entities["robot"]
    robot.articulation = copy.deepcopy(robot.articulation)
    assert robot.articulation is not None
    for actuator in robot.articulation.actuators:
      assert isinstance(actuator, BuiltinPositionActuatorCfg)
      actuator.stiffness *= gain_scale
      actuator.damping *= gain_scale
  return cfg
