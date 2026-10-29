"""Physical calibration, frame semantics, symmetry, and task isolation checks."""

import math
from types import SimpleNamespace
from typing import cast

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.pikachu.pikachu_v033_constants import get_spec
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.pikachu_v033.symmetry import (
  _JOINT_NAMES,
  _mirror_action,
  _mirror_named,
)
from mjlab.tasks.velocity.config.pikachu_v033_crawl import mdp
from mjlab.tasks.velocity.config.pikachu_v033_crawl.robot import (
  HEIGHT,
  JOINT_POSE,
  PITCH,
  robot_cfg,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
)


def calibrated_model():
  model = Entity(robot_cfg()).compile()
  data = mujoco.MjData(model)
  data.qpos[:3] = [0, 0, HEIGHT]
  data.qpos[3:7] = [math.cos(PITCH / 2), 0, math.sin(PITCH / 2), 0]
  for name, value in JOINT_POSE.items():
    data.qpos[model.joint(name).qposadr[0]] = value
  mujoco.mj_forward(model, data)
  return model, data


def test_crawl_four_bar_closure_and_joint_limits():
  model, data = calibrated_model()
  for side in ("right", "left"):
    error = (
      data.site(f"{side}_knee_mid_closure").xpos - data.site(f"{side}_knee_mid").xpos
    )
    assert np.linalg.norm(error) < 1e-6
  for j in range(1, model.njnt):
    if model.jnt_limited[j]:
      assert (
        model.jnt_range[j, 0]
        <= data.qpos[model.jnt_qposadr[j]]
        <= model.jnt_range[j, 1]
      )


def test_crawl_user_pose_geometry_and_mass():
  model, data = calibrated_model()
  for side in ("right", "left"):
    assert abs(data.site(f"{side}_foot").xpos[2]) < 1e-5
    assert f"{side}_hand_collision" not in (
      model.geom(i).name for i in range(model.ngeom)
    )
    mesh = model.geom(f"{side}_elbow_link_collision")
    assert mesh.type[0] == mujoco.mjtGeom.mjGEOM_MESH
    assert mesh.contype[0] == mesh.conaffinity[0] == 1
    # Reference sites now lie on the actual STL support surface, without the
    # old sphere-radius offset in the swing-height reward.
    mesh_id = int(mesh.dataid[0])
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start : start + count]
    world = vertices @ data.geom(mesh.id).xmat.reshape(3, 3).T + data.geom(mesh.id).xpos
    assert math.isclose(
      data.site(f"{side}_hand").xpos[2], world[:, 2].min(), abs_tol=1e-6
    )
  np.testing.assert_allclose(
    data.site("left_hand").xpos,
    data.site("right_hand").xpos * [1, -1, 1],
    atol=1e-6,
  )
  com = data.subtree_com[model.body("base_link").id]
  assert -0.10 < com[0] < 0.24
  assert abs(com[1]) < 0.10
  assert math.isclose(model.body_mass.sum(), 4.702736, abs_tol=1e-5)


def test_crawl_frames_are_invariant_under_world_yaw():
  yaw = torch.tensor([0.0, 0.7, 2.0, -2.5])
  zero = torch.zeros_like(yaw)
  q_yaw = quat_from_euler_xyz(zero, zero, yaw)
  tilt = quat_from_euler_xyz(zero, zero + PITCH, zero)
  q = quat_mul(q_yaw, tilt)
  velocity = torch.tensor([[0.15, 0.03, 0.0]]).expand(4, -1)
  velocity_w = quat_apply(q_yaw, velocity)
  gravity = quat_apply_inverse(q, torch.tensor([[0.0, 0.0, -1.0]]).expand(4, -1))
  data = SimpleNamespace(
    root_link_quat_w=q, root_link_lin_vel_w=velocity_w, projected_gravity_b=gravity
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}),
  )
  torch.testing.assert_close(mdp.heading_lin_vel(env), velocity, atol=1e-6, rtol=1e-6)
  torch.testing.assert_close(mdp.orientation_error(env), zero, atol=1e-6, rtol=0)


def test_all_v033_crawl_play_tasks_record_full_metrics_without_training_overhead():
  for suffix in ("Crawl", "Crawl-E1", "Crawl-E2"):
    task = f"Mjlab-Velocity-Flat-Pikachu-V033-{suffix}"
    training = load_env_cfg(task, play=False)
    playback = load_env_cfg(task, play=True)
    assert "support_count" in training.metrics
    assert "foot_tilt_deg/right" not in training.metrics
    for name in (
      "support_count",
      "vel/cmd_x",
      "vel/act_x",
      "vel/err_x",
      "joint_position_rad/right_ankle_joint",
      "foot_tilt_deg/right",
      "foot_contact/right",
      "foot_force_normal_n/right",
      "support_contact/right_front",
      "support_force_normal_n/left_rear",
    ):
      assert name in playback.metrics
    assert len(playback.metrics) > 200
    assert playback.metrics["vel/act_x"].func is mdp.heading_velocity_metric
    assert playback.metrics["foot_slip_velocity_m_s"].func is (
      mdp.rear_foot_contact_speed_metric
    )


def test_crawl_metrics_use_horizontal_heading_and_rear_contacts_only():
  quat = torch.tensor([[math.cos(PITCH / 2), 0.0, math.sin(PITCH / 2), 0.0]])
  data = SimpleNamespace(
    root_link_quat_w=quat,
    root_link_lin_vel_w=torch.tensor([[0.10, 0.02, 0.0]]),
    root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 0.03]]),
    site_lin_vel_w=torch.tensor([[[0.04, 0.0, 0.0], [0.25, 0.0, 0.0]]]),
  )
  robot = SimpleNamespace(
    data=data,
    find_sites=lambda names, preserve_order: ([0, 1], names),
  )
  sensor = SimpleNamespace(
    primary_names=(
      "right_ankle_link",
      "left_ankle_link",
      "right_elbow_link",
      "left_elbow_link",
    ),
    data=SimpleNamespace(found=torch.tensor([[1.0, 0.0, 1.0, 1.0]])),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": robot, "feet_ground_contact": sensor},
      command_manager=SimpleNamespace(
        get_command=lambda name: torch.tensor([[0.15, 0.0, 0.05]])
      ),
    ),
  )
  torch.testing.assert_close(
    mdp.heading_velocity_error_metric(env, 0), torch.tensor([0.05])
  )
  torch.testing.assert_close(
    mdp.rear_foot_contact_speed_metric(env), torch.tensor([0.25])
  )
  torch.testing.assert_close(
    mdp.rear_foot_contact_speed_metric(env, threshold=0.2), torch.tensor([1.0])
  )
  torch.testing.assert_close(
    mdp.support_contact_metric(env, "right_elbow_link"), torch.tensor([1.0])
  )


def test_biped_mirror_is_involution_for_all_eighteen_actions():
  action = torch.randn(32, 18, generator=torch.Generator().manual_seed(42))
  torch.testing.assert_close(_mirror_action(_mirror_action(action)), action)


def test_biped_joint_mirror_matches_physical_forward_kinematics():
  model = get_spec().compile()
  original, reflected = mujoco.MjData(model), mujoco.MjData(model)
  assert tuple(model.joint(i).name for i in range(1, model.njnt)) == _JOINT_NAMES
  samples = torch.randn(8, 22, generator=torch.Generator().manual_seed(42)) * 0.1
  for sample in samples:
    original.qpos[7:] = sample.numpy()
    reflected.qpos[7:] = _mirror_named(sample, _JOINT_NAMES).numpy()
    mujoco.mj_forward(model, original)
    mujoco.mj_forward(model, reflected)
    for body in range(1, model.nbody):
      name = model.body(body).name
      partner = (
        name.replace("right_", "left_")
        if name.startswith("right_")
        else name.replace("left_", "right_")
      )
      expected = original.xpos[body] * np.array([1.0, -1.0, 1.0])
      np.testing.assert_allclose(
        reflected.xpos[model.body(partner).id], expected, atol=3e-6
      )
      reflection = np.diag([1.0, -1.0, 1.0])
      np.testing.assert_allclose(
        reflected.xmat[model.body(partner).id].reshape(3, 3),
        reflection @ original.xmat[body].reshape(3, 3) @ reflection,
        atol=3e-6,
      )


def test_crawl_heading_stays_continuous_past_vertical_pitch():
  yaw = torch.tensor([0.7, 0.7, 0.7, 0.7])
  pitch = torch.tensor([math.radians(x) for x in (75, 85, 95, 110)])
  zeros = torch.zeros_like(yaw)
  rotation = quat_mul(
    quat_from_euler_xyz(zeros, zeros, yaw),
    quat_from_euler_xyz(zeros, pitch, zeros),
  )
  forward = torch.tensor([[1.0, 0.0, 0.0]]).expand(4, -1)
  actual = quat_apply(mdp.crawl_heading(rotation), forward)
  expected = quat_apply(quat_from_euler_xyz(zeros, zeros, yaw), forward)
  torch.testing.assert_close(actual, expected)


def test_crawl_phase_lifts_at_most_one_limb_and_gates_standing():
  command = torch.tensor([[0.15, 0.0, 0.0]]).expand(1600, -1).clone()
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      episode_length_buf=torch.arange(1600),
      step_dt=0.001,
      command_manager=SimpleNamespace(get_command=lambda name: command),
    ),
  )
  mask = mdp.swing_mask(env)
  assert mask.sum(-1).max() == 1
  torch.testing.assert_close(
    mask.float().mean(0), torch.full((4,), 0.2), atol=0.001, rtol=0
  )
  command.zero_()
  assert not mdp.swing_mask(env).any()
  assert not mdp.phase_observation(env).any()


def test_task_factories_and_runners_are_isolated():
  baseline = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-E0")
  crawl = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl")
  crawl.scene.entities["robot"].init_state.pos = (0, 0, 123)
  assert baseline.scene.entities["robot"].init_state.pos[2] == 0.259412
  assert baseline.rewards["base_height"].params["target_height"] == 0.259412
  assert baseline.events["reset_base"].params["pose_range"]["z"] == (0.0, 0.0)
  biped_play = load_env_cfg("Mjlab-Velocity-Flat-Pikachu-V033-E0", play=True)
  assert biped_play.events["reset_base"].params["pose_range"]["z"] == (0.0, 0.0)
  biped_rl = load_rl_cfg("Mjlab-Velocity-Flat-Pikachu-V033-E0")
  crawl_rl = load_rl_cfg("Mjlab-Velocity-Flat-Pikachu-V033-Crawl")
  assert isinstance(biped_rl, RslRlOnPolicyRunnerCfg)
  assert isinstance(crawl_rl, RslRlOnPolicyRunnerCfg)
  assert biped_rl.algorithm.symmetry_cfg is not None
  assert biped_rl.wandb_project != crawl_rl.wandb_project
  assert biped_rl.algorithm.symmetry_cfg["mirror_loss_coeff"] == 0.05
  assert crawl_rl.algorithm.symmetry_cfg is None
  robot = Entity(crawl.scene.entities["robot"])
  action = crawl.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  _, controlled = robot.find_joints_by_actuator_names(action.actuator_names)
  assert len(controlled) == 18
  assert sum("elbow" in name for name in controlled) == 2
  pose = robot.cfg.init_state.joint_pos
  assert pose is not None
  for side in ("right", "left"):
    assert pose[f"{side}_elbow_joint"] == 0
    assert pose[f"{side}_arm_yaw_joint"] == math.pi / 2
