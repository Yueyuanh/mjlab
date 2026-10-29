"""V035 ``arm_roll`` axes point along -x, so the default pose must hang arms down.

V033/V034 use +x. Copying their L-53/R+53 default to V035 raises the upper arm
53 degrees above the horizontal, which renders as "hands raised", and the
absolute ``arm_position_window`` bounds reward the same pose.
"""

import math

import mujoco
import pytest

from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import (
  BIPED_KEYFRAME,
  PIKACHU_V035_XML,
)
from mjlab.tasks.velocity.config.pikachu_v035_biped_e1.env_cfgs import (
  v035_biped_e1_env_cfg,
)

SIDES = (("left", 1.0), ("right", -1.0))


@pytest.fixture(scope="module")
def stand_pose():
  model = mujoco.MjModel.from_xml_path(str(PIKACHU_V035_XML))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
  mujoco.mj_forward(model, data)
  return model, data


@pytest.mark.parametrize("side, sign", SIDES)
def test_arm_roll_uses_the_negative_x_axis(stand_pose, side, sign):
  model, _ = stand_pose
  assert model.joint(f"{side}_arm_roll_joint").axis[0] == -1.0
  joint_pos = BIPED_KEYFRAME.joint_pos
  assert joint_pos is not None
  assert math.copysign(1.0, joint_pos[f"{side}_arm_roll_joint"]) == sign


@pytest.mark.parametrize("side, sign", SIDES)
def test_mjcf_keyframe_matches_the_initial_state(stand_pose, side, sign):
  model, data = stand_pose
  address = model.joint(f"{side}_arm_roll_joint").qposadr[0]
  joint_pos = BIPED_KEYFRAME.joint_pos
  assert joint_pos is not None
  expected = joint_pos[f"{side}_arm_roll_joint"]
  assert data.qpos[address] == pytest.approx(expected, abs=1e-12)
  assert math.copysign(1.0, data.qpos[address]) == sign


@pytest.mark.parametrize("side", ("left", "right"))
def test_hands_hang_below_the_shoulders(stand_pose, side):
  _, data = stand_pose
  shoulder = data.body(f"{side}_arm_pitch_link").xpos[2]
  elbow = data.body(f"{side}_elbow_link").xpos[2]
  # The arm segment is 10.55 cm; 53 degrees drops the distal link ~8.4 cm.
  assert elbow < shoulder - 0.05


def test_arm_window_points_the_same_way_as_the_default_pose():
  cfg = v035_biped_e1_env_cfg(play=True)
  bounds = cfg.rewards["arm_position_window"].params["bounds_degrees"]
  # ARM_JOINTS lists the right arm before the left, so roll sits at 1 and 5.
  assert bounds[1] == (-53.0, -45.0)
  assert bounds[5] == (45.0, 53.0)
  default = cfg.scene.entities["robot"].init_state.joint_pos
  assert default is not None
  assert default["left_arm_roll_joint"] > 0 > default["right_arm_roll_joint"]
