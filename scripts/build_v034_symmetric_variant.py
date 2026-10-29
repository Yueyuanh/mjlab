"""Build a V034 variant whose mirrored left-leg limits are physically mirrored.

The CAD export gives the left knee/ankle joints the same numerical ranges as the
right ones even though their axes point the opposite way.  A mirrored hinge
``q_l = -q_r`` must therefore use ``[-upper, -lower]``.  Without this correction
the left passive knee can flex only 0.65 rad (37 deg) instead of 1.05 rad
(60 deg), cutting the left foot's vertical stroke from 9.6 cm to 5.6 cm.

The original asset and tests are left untouched.  New tasks opt into this file
through ``get_biped_robot_cfg(symmetric=True)``.
"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/mjlab/asset_zoo/robots/pikachu/xmls/pikachu_v034"
XML_REPLACEMENTS = (
  (
    'name="left_knee_joint" pos="0 0 0" axis="0 -1 0" limited="true" '
    'range="-0.65 1.05"',
    'name="left_knee_joint" pos="0 0 0" axis="0 -1 0" limited="true" '
    'range="-1.05 0.65"',
  ),
  (
    'name="left_ankle_joint" pos="0 0 0" axis="0 -1 0" limited="true" '
    'range="-0.79 1.05"',
    'name="left_ankle_joint" pos="0 0 0" axis="0 -1 0" limited="true" '
    'range="-1.05 0.79"',
  ),
  (
    'name="left_knee_up_joint" pos="0 0 0" axis="0 -1 0" limited="true" '
    'range="-1.5707963267948966 1.2217304763960306"',
    'name="left_knee_up_joint" pos="0 0 0" axis="0 -1 0" limited="true" '
    'range="-1.2217304763960306 1.5707963267948966"',
  ),
)
URDF_JOINTS = {
  "left_knee_joint": ("-0.65", "1.05", "-1.05", "0.65"),
  "left_ankle_joint": ("-0.79", "1.05", "-1.05", "0.79"),
  "left_knee_up_joint": (
    "-1.5707963267948966",
    "1.2217304763960306",
    "-1.2217304763960306",
    "1.5707963267948966",
  ),
}


def build_xml(source: Path, destination: Path) -> None:
  text = source.read_text()
  for old, new in XML_REPLACEMENTS:
    if text.count(old) != 1:
      raise ValueError(f"{source.name}: expected exactly one {old!r}")
    text = text.replace(old, new)
  destination.write_text(text)


def build_urdf(source: Path, destination: Path) -> None:
  lines = source.read_text().splitlines(keepends=True)
  seen: set[str] = set()
  for index, line in enumerate(lines):
    for name, (old_lo, old_hi, new_lo, new_hi) in URDF_JOINTS.items():
      if f'name="{name}"' not in line:
        continue
      seen.add(name)
      # The <limit> element is within the next few lines of the joint header.
      for offset in range(index, min(index + 14, len(lines))):
        lines[offset] = (
          lines[offset]
          .replace(f'lower="{old_lo}"', f'lower="{new_lo}"')
          .replace(f'upper="{old_hi}"', f'upper="{new_hi}"')
        )
  if seen != set(URDF_JOINTS):
    raise ValueError(f"{source.name}: missing joints {set(URDF_JOINTS) - seen}")
  destination.write_text("".join(lines))


def foot_strokes(model: mujoco.MjModel, side: str) -> tuple[float, float]:
  """Return (foot z min, max) over the usable knee_up range.

  MuJoCo's equality constraint is enforced by the solver, not by ``mj_forward``;
  solve the two passive joints explicitly so the foot position is physical.
  """
  from scipy.optimize import least_squares

  data = mujoco.MjData(model)
  knee_up = model.joint(f"{side}_knee_up_joint")
  knee = model.joint(f"{side}_knee_joint")
  knee_mid = model.joint(f"{side}_knee_mid_joint")
  knee_up_range = model.jnt_range[knee_up.id]
  knee_range = model.jnt_range[knee.id]
  passive = [knee.qposadr[0], knee_mid.qposadr[0]]
  closure_a = model.site(f"{side}_knee_mid_closure").id
  closure_b = model.site(f"{side}_knee_mid").id

  def residual(value):
    data.qpos[passive] = value
    mujoco.mj_forward(model, data)
    return data.site_xpos[closure_a] - data.site_xpos[closure_b]

  heights = []
  for q in np.linspace(knee_up_range[0], knee_up_range[1], 241):
    mujoco.mj_resetData(model, data)
    data.qpos[knee_up.qposadr[0]] = q
    fit = least_squares(residual, [0.0, 0.0], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if np.linalg.norm(fit.fun) > 1e-7:
      continue
    passive_value = fit.x[0]
    if not knee_range[0] - 1e-6 <= passive_value <= knee_range[1] + 1e-6:
      continue
    heights.append(float(data.site(f"{side}_foot").xpos[2]))
  return min(heights), max(heights)


def main() -> None:
  xml_sources = (
    ASSET / "mjcf/pikachu_v034.xml",
    ASSET / "mjcf/pikachu_v034_train.xml",
  )
  hashes: dict[str, str] = {}
  for source in xml_sources:
    destination = source.with_name(source.stem + "_symmetric.xml")
    build_xml(source, destination)
    hashes[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(destination)
  urdf_source = ASSET / "urdf/pikachu_v034.urdf"
  urdf_destination = ASSET / "urdf/pikachu_v034_symmetric.urdf"
  build_urdf(urdf_source, urdf_destination)
  hashes[urdf_destination.name] = hashlib.sha256(
    urdf_destination.read_bytes()
  ).hexdigest()

  original = mujoco.MjModel.from_xml_path(str(ASSET / "mjcf/pikachu_v034_train.xml"))
  symmetric = mujoco.MjModel.from_xml_path(
    str(ASSET / "mjcf/pikachu_v034_train_symmetric.xml")
  )
  original_strokes = [foot_strokes(original, side) for side in ("right", "left")]
  symmetric_strokes = [foot_strokes(symmetric, side) for side in ("right", "left")]
  original_span = [hi - lo for lo, hi in original_strokes]
  symmetric_span = [hi - lo for lo, hi in symmetric_strokes]
  print("original foot z span", original_span)
  print("symmetric foot z span", symmetric_span)
  if abs(symmetric_span[0] - symmetric_span[1]) > 1e-6:
    raise ValueError("symmetric variant still has unequal foot strokes")
  if min(symmetric_span) < 0.09:
    raise ValueError("symmetric variant did not recover the full stroke")
  manifest = {
    "reason": "mirror left-leg hinge limits under axis reversal",
    "source": {
      "pikachu_v034.xml": hashlib.sha256(xml_sources[0].read_bytes()).hexdigest(),
      "pikachu_v034_train.xml": hashlib.sha256(xml_sources[1].read_bytes()).hexdigest(),
      "pikachu_v034.urdf": hashlib.sha256(urdf_source.read_bytes()).hexdigest(),
    },
    "variant_sha256": hashes,
    "foot_z_span_m": {
      "original_right_left": original_span,
      "symmetric_right_left": symmetric_span,
    },
  }
  out = ASSET / "provenance/symmetric_limits.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(manifest, indent=2) + "\n")
  print(out)


if __name__ == "__main__":
  main()
