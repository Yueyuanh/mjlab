"""Mirrored left-leg limits keep both V034 legs mechanically equivalent."""

import mujoco
import numpy as np
import pytest
from scipy.optimize import least_squares

from mjlab.asset_zoo.robots.pikachu.pikachu_v034_constants import (
  get_spec,
  get_symmetric_spec,
)

MIRRORED = ("knee_joint", "knee_up_joint", "ankle_joint")


@pytest.mark.parametrize("kind", MIRRORED)
def test_left_limits_mirror_right(kind):
  original = get_spec().compile()
  symmetric = get_symmetric_spec().compile()
  right = f"right_{kind}"
  left = f"left_{kind}"
  # Right side is untouched, left flips sign and swaps the bounds.
  np.testing.assert_allclose(
    symmetric.jnt_range[original.joint(right).id],
    original.jnt_range[original.joint(right).id],
  )
  low, high = original.jnt_range[original.joint(left).id]
  np.testing.assert_allclose(
    symmetric.jnt_range[symmetric.joint(left).id], [-high, -low], atol=1e-9
  )


def test_left_axis_is_reversed():
  model = get_symmetric_spec().compile()
  for kind in MIRRORED:
    np.testing.assert_allclose(model.joint(f"right_{kind}").axis, [0, 1, 0])
    np.testing.assert_allclose(model.joint(f"left_{kind}").axis, [0, -1, 0])


def _foot_height(model, side, knee_up):
  """Foot z with the four-bar closure solved by hand (mj_forward ignores it)."""
  data = mujoco.MjData(model)
  knee = model.joint(f"{side}_knee_joint")
  knee_mid = model.joint(f"{side}_knee_mid_joint")
  passive = [knee.qposadr[0], knee_mid.qposadr[0]]
  a = model.site(f"{side}_knee_mid_closure").id
  b = model.site(f"{side}_knee_mid").id

  def residual(value):
    data.qpos[passive] = value
    mujoco.mj_forward(model, data)
    return data.site_xpos[a] - data.site_xpos[b]

  mujoco.mj_resetData(model, data)
  data.qpos[model.joint(f"{side}_knee_up_joint").qposadr[0]] = knee_up
  fit = least_squares(residual, [0.0, 0.0], xtol=1e-12, ftol=1e-12, gtol=1e-12)
  if np.linalg.norm(fit.fun) > 1e-7:
    return None
  if not model.jnt_range[knee.id][0] <= fit.x[0] <= model.jnt_range[knee.id][1]:
    return None
  return float(data.site(f"{side}_foot").xpos[2])


def _stroke(model, side, samples=61):
  low, high = model.jnt_range[model.joint(f"{side}_knee_up_joint").id]
  heights = [
    z
    for z in (_foot_height(model, side, q) for q in np.linspace(low, high, samples))
    if z is not None
  ]
  return max(heights) - min(heights)


def test_symmetric_variant_equalizes_foot_stroke():
  original = get_spec().compile()
  symmetric = get_symmetric_spec().compile()
  # The CAD export's left leg travels visibly less in the original asset.
  original_right = _stroke(original, "right")
  original_left = _stroke(original, "left")
  assert original_left < original_right - 0.02
  # Mirroring the limits restores equality.
  np.testing.assert_allclose(
    _stroke(symmetric, "left"), _stroke(symmetric, "right"), atol=1e-4
  )


def test_knee_flexion_direction():
  model = get_symmetric_spec().compile()
  # Standing pose is the most extended configuration; flexion shortens the leg.
  for side, sign in (("right", 1.0), ("left", -1.0)):
    straight = _foot_height(model, side, 0.0)
    flexed = _foot_height(model, side, sign * 0.5)
    assert flexed is not None and straight is not None
    assert flexed > straight + 0.02
    # Negative-sign values are the other branch of the four-bar (extension).
    extension = _foot_height(model, side, -sign * 0.5)
    assert extension is not None and extension < straight
