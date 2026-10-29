"""Reordering changes indices, never V034's physical model or named poses."""

import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import ASSET_DIR
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg

ORIGINAL = ASSET_DIR / "provenance/pre_v033_order"
REFERENCE = ASSET_DIR.parent / "pikachu_v033/mjcf/pikachu_v033_train.xml"


def _canonical(node):
  attributes = dict(node.attrib)
  if node.tag == "key":
    # State vectors are checked by joint name on the compiled models below.
    attributes.pop("qpos", None)
    attributes.pop("qvel", None)
  children = list(node)
  children = [c for c in children if c.tag == "freejoint"] + [
    c for c in children if c.tag != "freejoint"
  ]
  tag = "mesh" if node.tag == "asset" else "body"
  slots = [i for i, child in enumerate(children) if child.tag == tag]
  ordered = sorted((children[i] for i in slots), key=lambda c: c.get("name", ""))
  for i, child in zip(slots, ordered, strict=True):
    children[i] = child
  return node.tag, attributes, [_canonical(child) for child in children]


def _original_model(name):
  root = ET.parse(ORIGINAL / name).getroot()
  compiler = root.find("compiler")
  assert compiler is not None
  compiler.set("meshdir", str(ASSET_DIR / "meshes"))
  return mujoco.MjModel.from_xml_string(
    ET.tostring(root, encoding="unicode"),
    assets={"scene.xml": (ORIGINAL / "scene.xml").read_bytes()},
  )


@pytest.mark.parametrize("name", ["pikachu_v034.xml", "pikachu_v034_train.xml"])
def test_xml_order_and_native_attributes(name):
  original = ET.parse(ORIGINAL / name).getroot()
  actual = ET.parse(ASSET_DIR / "mjcf" / name).getroot()
  reference = ET.parse(REFERENCE).getroot()
  for tag in ("body", "joint", "mesh"):
    expected = [node.get("name") for node in reference.iter(tag) if node.get("name")]
    # Ignore unnamed <default><joint> templates.
    assert [
      node.get("name") for node in actual.iter(tag) if node.get("name")
    ] == expected
  assert _canonical(actual) == _canonical(original)


@pytest.mark.parametrize("name", ["pikachu_v034.xml", "pikachu_v034_train.xml"])
def test_named_keyframes_and_kinematics_unchanged(name):
  original = _original_model(name)
  actual = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "mjcf" / name))
  assert (actual.nq, actual.nv, actual.nu, actual.neq) == (
    original.nq,
    original.nv,
    original.nu,
    original.neq,
  )
  np.testing.assert_allclose(actual.body_mass.sum(), original.body_mass.sum())
  old_data, new_data = mujoco.MjData(original), mujoco.MjData(actual)
  for key in range(-1, original.nkey):
    if key >= 0:
      old_key = original.key(key)
      new_key = actual.key(old_key.name)
      mujoco.mj_resetDataKeyframe(original, old_data, old_key.id)
      mujoco.mj_resetDataKeyframe(actual, new_data, new_key.id)
      for i in range(original.njnt):
        old_joint = original.joint(i)
        new_joint = actual.joint(old_joint.name)
        width = 7 if old_joint.type[0] == mujoco.mjtJoint.mjJNT_FREE else 1
        a, b = old_joint.qposadr[0], new_joint.qposadr[0]
        np.testing.assert_array_equal(
          old_data.qpos[a : a + width], new_data.qpos[b : b + width]
        )
    else:
      # A nonzero pose catches accidental reparenting and joint-axis changes.
      for i in range(original.njnt):
        joint = original.joint(i)
        if joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE:
          continue
        value = 0.12 * np.sin(i + 1)
        old_data.qpos[joint.qposadr[0]] = value
        new_data.qpos[actual.joint(joint.name).qposadr[0]] = value
    mujoco.mj_forward(original, old_data)
    mujoco.mj_forward(actual, new_data)
    for i in range(1, original.nbody):
      j = actual.body(original.body(i).name).id
      np.testing.assert_allclose(old_data.xpos[i], new_data.xpos[j], atol=1e-12)
      np.testing.assert_allclose(old_data.xmat[i], new_data.xmat[j], atol=1e-12)


def test_original_hashes_and_scene_preserved():
  hashes = json.loads((ORIGINAL / "sha256.json").read_text())
  for name, digest in hashes.items():
    assert hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest() == digest
  manifest = json.loads((ASSET_DIR / "source_manifest.json").read_text())
  assert hashes["pikachu_v034.xml"] == manifest["sha256"]["mjcf/pikachu_v034.xml"]
  assert (ASSET_DIR / "mjcf/scene.xml").read_bytes() == (
    ORIGINAL / "scene.xml"
  ).read_bytes()


@pytest.mark.parametrize("mode", ["Biped", "Crawl", "Biped-E1", "Crawl-E3"])
def test_contact_reward_sites_follow_sensor_body_order(mode):
  cfg = load_env_cfg(f"Mjlab-Velocity-Flat-Pikachu-V034-{mode}")
  robot = Entity(cfg.scene.entities["robot"])
  crawl = mode.startswith("Crawl")
  _, bodies = robot.find_bodies(
    r"(left|right)_(ankle|elbow)_link" if crawl else r"(left|right)_ankle_link"
  )
  sites = tuple(
    name.replace("_ankle_link", "_foot").replace("_elbow_link", "_hand")
    for name in bodies
  )
  term = cfg.rewards["support_slip" if crawl else "foot_slip"]
  asset = term.params["asset_cfg"]
  _, actual_sites = robot.find_sites(
    asset.site_names, preserve_order=asset.preserve_order
  )
  assert tuple(actual_sites) == sites
  if not crawl:
    scan = next(s for s in cfg.scene.sensors if s.name == "foot_height_scan")
    assert tuple(frame.name for frame in scan.frame) == sites
