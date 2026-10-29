"""V034 asset invariants that catch accidental V033 conventions and actuator drift."""

import json

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  ASSET_DIR,
  JOINT_NAMES,
  PASSIVE_JOINTS,
  get_spec,
)
from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


def test_native_parameters_preserved():
  source = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "mjcf/pikachu_v034.xml"))
  train = get_spec().compile()
  assert train.nq == 29 and train.nv == 28 and train.nu == 0 and train.neq == 2
  assert train.body_mass.sum() == pytest.approx(source.body_mass.sum(), abs=1e-8)
  for i in range(source.njnt):
    original = source.joint(i)
    actual = train.joint(original.name)
    np.testing.assert_allclose(actual.axis, original.axis)
    np.testing.assert_allclose(actual.range, original.range)
    assert actual.limited == original.limited


@pytest.mark.parametrize("mode", ["Biped", "Crawl"])
def test_registered_asset_and_reset(mode):
  task = f"Mjlab-Velocity-Flat-Pikachu-V034-{mode}"
  assert task in list_tasks()
  cfg = load_env_cfg(task)
  robot = Entity(cfg.scene.entities["robot"])
  model = robot.spec.compile()
  assert model.nu == 18
  driven = {model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)}
  assert driven == set(JOINT_NAMES)
  assert not driven.intersection(PASSIVE_JOINTS)
  for i in range(model.nu):
    joint = model.joint(int(model.actuator_trnid[i, 0])).name
    effort = 3.0 if "arm_yaw" in joint or "elbow" in joint else 12.5
    np.testing.assert_allclose(model.actuator_forcerange[i], [-effort, effort])
  assert model.nkey > 0
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  mujoco.mj_forward(model, data)
  np.testing.assert_allclose(data.qpos[:3], cfg.scene.entities["robot"].init_state.pos)
  np.testing.assert_allclose(data.qpos[3:7], cfg.scene.entities["robot"].init_state.rot)
  for side in ("left", "right"):
    error = np.linalg.norm(
      data.site(f"{side}_knee_mid_closure").xpos - data.site(f"{side}_knee_mid").xpos
    )
    assert error < 1e-5
  assert load_rl_cfg(task).algorithm.symmetry_cfg is None
  cfg.scene.num_envs = 7
  assert load_env_cfg(task).scene.num_envs != 7


def test_crawl_geometry_support_and_limits():
  from mjlab.tasks.velocity.config.pikachu_v034.robot import CALIBRATION, crawl_spec

  model = crawl_spec().compile()
  data = mujoco.MjData(model)
  pitch = CALIBRATION["pitch"]
  data.qpos[:7] = [
    0,
    0,
    CALIBRATION["height"],
    np.cos(pitch / 2),
    0,
    np.sin(pitch / 2),
    0,
  ]
  for name, value in CALIBRATION["joints"].items():
    joint = model.joint(name)
    data.qpos[joint.qposadr[0]] = value
    if joint.limited:
      assert joint.range[0] < value < joint.range[1]
  mujoco.mj_forward(model, data)
  for side in ("left", "right"):
    assert abs(data.site(f"{side}_foot").xpos[2]) < 1e-5
    assert data.site(f"{side}_hand").xpos[2] == pytest.approx(
      CALIBRATION["hand_radius"], abs=1e-5
    )
    normal = data.xmat[model.body(f"{side}_ankle_link").id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-4)
  com = data.subtree_com[model.body("base_link").id]
  assert -0.09 < com[0] < 0.24 and abs(com[1]) < 0.10


def test_snapshot_hashes():
  import hashlib

  manifest = json.loads((ASSET_DIR / "source_manifest.json").read_text())
  for path, digest in manifest["sha256"].items():
    digest = manifest.get("transformations", {}).get(path, {}).get("sha256", digest)
    assert hashlib.sha256((ASSET_DIR / path).read_bytes()).hexdigest() == digest


def test_crawl_contact_order_follows_names():
  import torch

  from mjlab.tasks.velocity.config.pikachu_v034 import crawl_mdp
  from mjlab.tasks.velocity.config.pikachu_v034.robot import crawl_spec

  # Resolve semantic support order even if model traversal changes again.
  model = crawl_spec().compile()
  names = [
    model.body(i).name
    for i in range(model.nbody)
    if model.body(i).name.endswith(("_ankle_link", "_elbow_link"))
  ]
  indices = crawl_mdp.support_contact_indices(names)
  assert [names[i] for i in indices] == [
    "right_ankle_link",
    "left_ankle_link",
    "right_elbow_link",
    "left_elbow_link",
  ]
  found = torch.tensor([[name != "right_ankle_link" for name in names]])
  assert found[:, indices].tolist() == [[False, True, True, True]]


def test_crawl_heading_stays_horizontal():
  import torch

  from mjlab.tasks.velocity.config.pikachu_v034.crawl_mdp import crawl_heading
  from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

  pitch = torch.tensor([1.4, np.pi / 2, 1.7])
  yaw = torch.full_like(pitch, 0.6)
  quat = quat_from_euler_xyz(torch.zeros_like(pitch), pitch, yaw)
  heading = crawl_heading(quat)
  forward = quat_apply(heading, torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1))
  torch.testing.assert_close(
    forward,
    torch.tensor([[np.cos(0.6), np.sin(0.6), 0.0]], dtype=torch.float32).expand(3, -1),
  )
