"""Asymmetric arm windows, stationary gating, mesh contact and task isolation."""

import math
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactSensor, ObjRef, TerrainHeightSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v033.symmetry import _mirror_named
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1 import mdp
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1.env_cfgs import (
  ARM_BOUNDS_DEGREES,
  ARM_JOINTS,
)
from mjlab.tasks.velocity.config.pikachu_v033_biped_e1.robot import (
  biped_e1_robot_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def arm_cost(angles: torch.Tensor) -> torch.Tensor:
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=angles))}
    ),
  )
  return mdp.arm_position_window(env, SceneEntityCfg("robot"), ARM_BOUNDS_DEGREES)


@pytest.mark.parametrize("joint", range(8))
@pytest.mark.parametrize("direction", (-1, 1))
def test_each_asymmetric_arm_boundary_and_excess(joint: int, direction: int):
  bounds = torch.deg2rad(torch.tensor(ARM_BOUNDS_DEGREES, dtype=torch.float64))
  poses = bounds.mean(-1).repeat(3, 1)
  edge = bounds[joint, 0 if direction == -1 else 1]
  half = (bounds[joint, 1] - bounds[joint, 0]) / 2
  poses[:, joint] = edge + direction * half * torch.arange(3)
  original = poses.clone()
  torch.testing.assert_close(
    arm_cost(poses),
    torch.tensor([0, 1 / 8, 4 / 8], dtype=torch.float64),
    atol=1e-12,
    rtol=1e-10,
  )
  torch.testing.assert_close(poses, original)


def test_keyframe_bounds_and_cost_are_physically_mirrored():
  robot = Entity(biped_e1_robot_cfg())
  model = robot.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  angles = torch.tensor(
    [[data.qpos[model.joint(n).qposadr[0]] for n in ARM_JOINTS]], dtype=torch.float64
  )
  np.testing.assert_allclose(
    torch.rad2deg(angles)[0], [0, 53, 0, 0, 0, -53, 0, 0], atol=1e-10
  )
  assert arm_cost(angles).item() == pytest.approx(0, abs=1e-12)
  reflected = _mirror_named(angles, ARM_JOINTS)
  torch.testing.assert_close(reflected, angles)
  probe = angles + torch.linspace(-0.2, 0.3, 8)
  torch.testing.assert_close(
    arm_cost(probe), arm_cost(_mirror_named(probe, ARM_JOINTS))
  )
  mujoco.mj_forward(model, data)
  assert data.ncon == 0  # no arm-leg overlap or proximity at reset
  assert model.body_mass.sum() == pytest.approx(4.7027358)


def test_contact_policy_only_enables_arms_against_legs_and_soles():
  model = Entity(biped_e1_robot_cfg()).compile()

  def enabled(a: str, b: str) -> bool:
    x, y = model.geom(a), model.geom(b)
    return bool((x.contype[0] & y.conaffinity[0]) or (y.contype[0] & x.conaffinity[0]))

  arm = "right_elbow_link_collision"
  assert enabled(arm, "right_knee_link_collision")
  assert enabled(arm, "left_hip_yaw_link_collision")
  assert enabled(arm, "left_foot1_collision")
  assert not enabled(arm, "left_elbow_link_collision")
  assert not enabled("right_knee_link_collision", "left_knee_link_collision")
  assert model.geom(arm).gap[0] == pytest.approx(0.01)
  assert model.geom(arm).margin[0] == 0
  assert all(
    "ankle_link_collision" not in model.geom(i).name for i in range(model.ngeom)
  )


def test_gap_zone_is_detected_without_generating_contact_force():
  # Exercise native MuJoCo semantics used by the proximity reward, including
  # inactive contacts. A version/backend change must not silently disable it.
  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="arm"><freejoint/><geom type="sphere" size=".02" gap=".01"/></body>
    <body pos=".045 0 0"><geom type="sphere" size=".02"/></body></worldbody>
    <sensor><contact name="near" body1="arm" data="found dist" reduce="mindist" num="1"/></sensor></mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.sensordata[0] == 1
  assert data.sensordata[1] == pytest.approx(0.005)
  assert data.contact[0].efc_address == -1


def quiet_env(command: torch.Tensor) -> ManagerBasedRlEnv:
  sensor = Mock(spec=ContactSensor)
  sensor.data = SimpleNamespace(found=torch.tensor([[1, 0], [1, 0]]))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      joint_vel=torch.ones(2, 18),
      root_link_lin_vel_w=torch.ones(2, 3) * 0.01,
      root_link_ang_vel_w=torch.ones(2, 3) * 0.02,
      site_lin_vel_w=torch.ones(2, 2, 3) * 0.01,
    )
  )
  return cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": robot, "feet": sensor},
      num_envs=2,
      device="cpu",
      command_manager=SimpleNamespace(get_command=lambda _: command),
    ),
  )


def test_quiet_costs_gate_on_command_not_actual_speed():
  env = quiet_env(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]))
  cfg = SceneEntityCfg("robot")
  for cost in (
    mdp.standing_joint_velocity(env, cfg),
    mdp.standing_base_motion(env),
    mdp.standing_foot_motion(env, cfg),
    mdp.standing_support(env, "feet"),
  ):
    assert cost[0] > 0
    assert cost[1] == 0  # pure turning is movement even with zero xy command
  with patch.object(mdp.common, "push_by_setting_velocity") as push:
    mdp.push_moving_robots(env, None, {"x": (-0.1, 0.1)})
    torch.testing.assert_close(push.call_args.args[1], torch.tensor([1]))


def test_mesh_clearance_cost_grows_before_penetration_and_ignores_missing():
  sensor = Mock(spec=ContactSensor)
  sensor.data = SimpleNamespace(
    found=torch.tensor([[1], [1], [1], [0]]),
    dist=torch.tensor([[0.01], [0.005], [-0.005], [0.0]]),
  )
  env = cast(ManagerBasedRlEnv, SimpleNamespace(scene={"near": sensor}))
  torch.testing.assert_close(
    mdp.arm_leg_clearance(env, "near"), torch.tensor([0.0, 0.25, 2.25, 0.0])
  )


def test_e1_keeps_action_model_and_isolates_reward_changes():
  e1 = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-E1")
  e0 = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-E0")
  action = e1.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  robot = Entity(e1.scene.entities["robot"])
  _, names = robot.find_joints_by_actuator_names(action.actuator_names)
  assert len(names) == 18
  baseline_action = e0.actions["joint_pos"]
  assert isinstance(baseline_action, JointPositionActionCfg)
  assert action.scale == baseline_action.scale
  assert action.clip is None
  rl = load_rl_cfg("Mjlab-Velocity-Flat-Pikachu-V033-E1")
  assert isinstance(rl, RslRlOnPolicyRunnerCfg)
  assert rl.algorithm.symmetry_cfg is not None
  assert rl.actor.hidden_dims == (512, 256, 128)
  assert rl.algorithm.symmetry_cfg["mirror_loss_coeff"] == 0.05
  assert rl.experiment_name == "pikahcu_v033/biped_e1"
  assert "arm_position_window" not in e0.rewards
  command, baseline_command = e1.commands["twist"], e0.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert isinstance(baseline_command, UniformVelocityCommandCfg)
  assert command.rel_standing_envs == 0.25
  assert baseline_command.rel_standing_envs == 0.10
  for sensor in e1.scene.sensors or ():
    if isinstance(sensor, TerrainHeightSensorCfg):
      assert isinstance(sensor.frame, tuple)
      frames = cast(tuple[ObjRef, ...], sensor.frame)
      assert tuple(f.name for f in frames) == ("right_foot", "left_foot")
  assert math.isclose(e1.scene.entities["robot"].init_state.pos[2], 0.259412)
