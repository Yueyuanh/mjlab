"""Import the V035 CAD export and build a reproducible training asset.

V035 has 18 direct-drive revolute joints (no four-bar knee, no passive joints).
The MJCF is produced by ``easymjcf`` from ``pikachu_v035.urdf``.  This script
turns that raw model into the training asset used by mjlab:

* strips the ``<include>``/``<size>``/``<actuator>``/``<keyframe>`` blocks,
* adds a floating base, an IMU site, and the standard sensors,
* tags visual/collision geoms and fits the foot capsule colliders,
* writes a ``stand`` keyframe at the CAD zero pose,
* records mass/height/feet bounds in ``geometry.json``.

Run from the mjlab repository root::

    uv run python scripts/build_v035_train_xml.py \
        --source /home/finnox/Pikachu/PikachuRobot/pikachu_urdf/pikachu_v035
"""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/mjlab/asset_zoo/robots/pikachu/xmls/pikachu_v035"

SOURCE_NAMES = (
  "mjcf/pikachu_v035.xml",
  "mjcf/scene.xml",
  "urdf/pikachu_v035.urdf",
  "config/robot_config.json",
  "config/symmetry_report.json",
)

# CAD zero pose: arms are rolled out 53 deg so they clear the legs.
# v035 的 arm_roll 轴是 (-1,0,0)（v033/v034 是 (+1,0,0)），所以把上臂压到
# 水平面下方 53° 需要 左 +53°/右 -53°；写反了就是"双手举起"。
ARM_ROLL_DEG = 53.0
FOOT_RADIUS = 0.009
FOOT_CAPSULES = 5


def mesh_vertices(model, data, body_name):
  body = model.body(body_name).id
  geom = next(
    g
    for g in range(model.ngeom)
    if model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
  )
  mesh = model.geom_dataid[geom]
  start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
  vertices = model.mesh_vert[start : start + count]
  world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
  return (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)


def import_source(source: Path) -> None:
  for folder in ("mjcf", "meshes", "urdf", "config"):
    (ASSET / folder).mkdir(parents=True, exist_ok=True)
  names = list(SOURCE_NAMES)
  names += [
    "meshes/" + mesh.attrib["file"]
    for mesh in ET.parse(source / "mjcf/pikachu_v035.xml").findall("./asset/mesh")
  ]
  hashes = {}
  for name in names:
    shutil.copy2(source / name, ASSET / name)
    hashes[name] = hashlib.sha256((ASSET / name).read_bytes()).hexdigest()
  manifest = {
    "source": str(source.resolve()),
    "sha256": hashes,
    "conversion": "easymjcf urdf/pikachu_v035.urdf",
    "transformations": {
      "mjcf/pikachu_v035_train.xml": (
        "freejoint + IMU site + sensors + foot capsules + stand keyframe"
      )
    },
  }
  (ASSET / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build(source: Path | None = None) -> dict:
  if source is not None:
    import_source(source)

  root = ET.parse(ASSET / "mjcf/pikachu_v035.xml").getroot()
  for tag in ("include", "size", "actuator", "keyframe"):
    for node in root.findall(tag):
      root.remove(node)

  default = ET.SubElement(root, "default")
  ET.SubElement(
    default, "joint", damping="0.3", frictionloss="0.01", solreflimit="0.004 1"
  )

  base = root.find(".//body[@name='base_link']")
  assert base is not None
  base.insert(0, ET.Element("freejoint", name="floating_base"))
  ET.SubElement(base, "site", name="imu", size="0.003")

  for body in root.findall(".//body"):
    for geom in body.findall("geom"):
      visual = geom.get("contype") == "0"
      geom.set("name", body.attrib["name"] + ("_visual" if visual else "_collision"))
      geom.set("group", "2" if visual else "3")

  dst = ASSET / "mjcf/pikachu_v035_train.xml"

  def write():
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(dst, encoding="unicode")

  write()
  model = mujoco.MjModel.from_xml_path(str(dst))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  geometry = {}
  for side in ("left", "right"):
    name = f"{side}_ankle_link"
    vertices = mesh_vertices(model, data, name)
    lo, hi = vertices.min(0), vertices.max(0)
    radius = FOOT_RADIUS
    z = float(lo[2])
    body = root.find(f".//body[@name='{name}']")
    assert body is not None
    ET.SubElement(
      body,
      "site",
      name=f"{side}_foot",
      pos=f"{(lo[0] + hi[0]) / 2} {(lo[1] + hi[1]) / 2} {z}",
      size="0.003",
    )
    for i, y in enumerate(
      np.linspace(lo[1] + radius, hi[1] - radius, FOOT_CAPSULES), 1
    ):
      ET.SubElement(
        body,
        "geom",
        name=f"{side}_foot{i}_collision",
        type="capsule",
        fromto=f"{lo[0] + radius} {y} {z + radius} {hi[0] - radius} {y} {z + radius}",
        size=str(radius),
        mass="0",
        group="3",
        condim="3",
        friction="0.8 0.005 0.0001",
        rgba="0.2 0.6 0.2 0.4",
      )
    geometry[side] = {"ankle_bounds": [lo.tolist(), hi.tolist()]}

  sensors = ET.SubElement(root, "sensor")
  for tag, name in (
    ("gyro", "imu_ang_vel"),
    ("velocimeter", "imu_lin_vel"),
    ("accelerometer", "imu_acc"),
  ):
    ET.SubElement(sensors, tag, name=name, site="imu")
  for tag, name in (
    ("framequat", "imu_quat"),
    ("framepos", "frame_pos"),
    ("framelinvel", "frame_vel"),
  ):
    ET.SubElement(sensors, tag, name=name, objtype="site", objname="imu")
  ET.SubElement(sensors, "subtreeangmom", name="root_angmom", body="base_link")
  write()

  model = mujoco.MjModel.from_xml_path(str(dst))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  height = -min(data.site(f"{s}_foot").xpos[2] for s in ("left", "right"))
  qpos = model.qpos0.copy()
  qpos[2] = height
  for side, sign in (("left", 1), ("right", -1)):
    qpos[model.joint(f"{side}_arm_roll_joint").qposadr[0]] = sign * np.deg2rad(
      ARM_ROLL_DEG
    )
  keyframe = ET.SubElement(root, "keyframe")
  ET.SubElement(keyframe, "key", name="stand", qpos=" ".join(map(str, qpos)))
  write()

  model = mujoco.MjModel.from_xml_path(str(dst))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  # Sanity: the keyframe must place the soles exactly on the ground.
  key = model.key("stand")
  data.qpos[:] = key.qpos
  mujoco.mj_forward(model, data)
  sole = min(data.site(f"{s}_foot").xpos[2] for s in ("left", "right"))
  result = {
    "mass_kg": float(model.body_mass.sum()),
    "stand_height": float(height),
    "keyframe_sole_z": float(sole),
    "nq": model.nq,
    "nv": model.nv,
    "equality_count": model.neq,
    "feet": geometry,
  }
  (ASSET / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  return result


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source", type=Path, help="V035 source directory; omit to rebuild snapshot"
  )
  build(parser.parse_args().source)
