"""Align V034 body/mesh traversal with V033 without changing physical values.

Only sibling bodies and mesh declarations move. Actuator/sensor declarations
retain their order; keyframe state vectors follow their named joints.
"""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/mjlab/asset_zoo/robots/pikachu/xmls/pikachu_v034"
REFERENCE = ASSET.parent / "pikachu_v033/mjcf/pikachu_v033_train.xml"


def joint_layout(root: ET.Element, velocity: bool = False) -> list[str]:
  """Return a semantic label for each qpos/qvel entry, including the free base."""
  widths = {"free": 6 if velocity else 7, "ball": 3 if velocity else 4}
  labels = []
  world = root.find("worldbody")
  assert world is not None
  for body in world.iter("body"):
    # MuJoCo assigns a body's joints before its children, even when the
    # freejoint declaration appears below child bodies in the XML.
    for node in body:
      if node.tag not in ("joint", "freejoint"):
        continue
      kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
      labels.extend(f"{node.attrib['name']}:{i}" for i in range(widths.get(kind, 1)))
  return labels


def align_tree(root: ET.Element) -> bool:
  reference = ET.parse(REFERENCE).getroot()
  ranks = {b.attrib["name"]: i for i, b in enumerate(reference.iter("body"))}
  before = {field: joint_layout(root, field == "qvel") for field in ("qpos", "qvel")}
  changed = False
  for parent in root.iter():
    free = parent.find("freejoint") if parent.tag == "body" else None
    if free is not None and parent[0] is not free:
      parent.remove(free)
      parent.insert(0, free)
      changed = True
    tag = "mesh" if parent.tag == "asset" else "body"
    slots = [i for i, child in enumerate(parent) if child.tag == tag]
    children = [parent[i] for i in slots]
    ordered = sorted(children, key=lambda child: ranks.get(child.get("name"), 10**6))
    changed |= children != ordered
    for i, child in zip(slots, ordered, strict=True):
      parent[i] = child
  for field, labels in before.items():
    after = joint_layout(root, field == "qvel")
    if after == labels:
      continue
    for key in root.findall("./keyframe/key"):
      if field not in key.attrib:
        continue
      values = key.attrib[field].split()
      mapping = dict(zip(labels, values, strict=True))
      key.set(field, " ".join(mapping[label] for label in after))
  return changed


def align_file(path: Path) -> None:
  tree = ET.parse(
    path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
  )
  if align_tree(tree.getroot()):
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode")


def update_manifest() -> None:
  """Keep imported hashes as provenance; record the normalized source separately."""
  path = ASSET / "source_manifest.json"
  manifest = json.loads(path.read_text())
  name = "mjcf/pikachu_v034.xml"
  digest = hashlib.sha256((ASSET / name).read_bytes()).hexdigest()
  if digest != manifest["sha256"][name]:
    manifest["transformations"] = {
      name: {
        "sha256": digest,
        "description": "V033 sibling body and mesh order; native attributes preserved",
      }
    }
  else:
    manifest.pop("transformations", None)
  path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
  backup = ASSET / "provenance/pre_v033_order"
  backup.mkdir(parents=True, exist_ok=True)
  hashes = {}
  for name in ("pikachu_v034.xml", "pikachu_v034_train.xml", "scene.xml"):
    src = ASSET / "mjcf" / name
    original = backup / name
    if not original.exists():
      shutil.copy2(src, original)
    hashes[name] = hashlib.sha256(original.read_bytes()).hexdigest()
    if name != "scene.xml":
      align_file(src)
  (backup / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
  update_manifest()


if __name__ == "__main__":
  main()
