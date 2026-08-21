"""Pikachu 四足全向 velocity 任务测试。

覆盖:
  1. 四足任务已注册(``Mjlab-Velocity-Flat-Pikachu-Quad``);
  2. 四足起步姿态 = 共享 ``QUAD_KEYFRAME``(手脚四点、base 前俯);
  3. 四足朝向奖励/终止被替换为 ``quad_upright``/``quad_bad_orientation``(带偏角);
  4. 四足动作缩放 = ``QUAD_ACTION_SCALE``、flat 平地、feet/self 传感器在;
  5. **共享任务/常量未变** —— 双足 Mjlab-Velocity-Flat-Pikachu 的
     upright(fall) 函数、action scale、base_height target 与四足前完全一致,
     共享 ``pikachu_constants`` 没有引入任何四足定义。
"""

import importlib

from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  QUAD_KEYFRAME,
)
from mjlab.asset_zoo.robots.pikachu.pikachu_quad_constants import (
  QUAD_ACTION_SCALE,
  QUAD_START,
  get_pikachu_quad_robot_cfg,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.config.pikachu_quad.quad_mdp import (
  QUAD_BODY_AXIS,
  QUAD_TILT_DEG,
  quad_bad_orientation,
  quad_upright,
)

QUAD_TASK = "Mjlab-Velocity-Flat-Pikachu-Quad"
BIPED_TASK = "Mjlab-Velocity-Flat-Pikachu"

# 共享模块对象(用于断言其中未被引入四足定义)。
_shared_constants = importlib.import_module(
  "mjlab.asset_zoo.robots.pikachu.pikachu_constants"
)


def test_quad_velocity_task_registered() -> None:
  """四足 velocity 任务应出现在任务表且可加载配置。"""
  assert QUAD_TASK in list_tasks(), f"{QUAD_TASK} missing from registry"
  cfg = load_env_cfg(QUAD_TASK)
  assert "twist" in cfg.commands


def test_quad_task_is_flat_plane() -> None:
  """四足起步任务是平面地形、无 height_scan 观测(去掉夜景地形传感器)。"""
  cfg = load_env_cfg(QUAD_TASK)
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["critic"].terms


def test_quad_start_is_shared_keyframe() -> None:
  """四足起步姿态复用共享 QUAD_KEYFRAME(手脚四点、base 前俯 90°)。"""
  quad_cfg = get_pikachu_quad_robot_cfg()
  assert QUAD_START == QUAD_KEYFRAME
  # Entity init_state 即 QUAD_START,与共享关键帧逐字段一致。
  assert quad_cfg.init_state == QUAD_KEYFRAME


def test_quad_upright_replaced() -> None:
  """四足 upright 奖励应替换为 quad_upright(带 tilt_deg 偏角)。"""
  cfg = load_env_cfg(QUAD_TASK)
  upright = cfg.rewards["upright"]
  assert isinstance(upright, RewardTermCfg)
  assert upright.func is quad_upright
  # 偏角参数已参数化(默认 90°,标注待门禁标定/QD 显式存在于模块)。
  assert upright.params["tilt_deg"] == QUAD_TILT_DEG
  assert upright.params["body_axis"] == tuple(QUAD_BODY_AXIS)


def test_quad_fell_over_replaced() -> None:
  """四足 fell_over 终止应替换为 quad_bad_orientation(带 tilt 偏角)。"""
  cfg = load_env_cfg(QUAD_TASK)
  fell = cfg.terminations["fell_over"]
  assert isinstance(fell, TerminationTermCfg)
  assert fell.func is quad_bad_orientation
  assert fell.params["tilt_deg"] == QUAD_TILT_DEG


def test_quad_action_scale() -> None:
  """四足动作缩放应为 QUAD_ACTION_SCALE(arm_pitch 前腿抬/落放宽)。"""
  cfg = load_env_cfg(QUAD_TASK)
  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  assert joint_pos.scale == QUAD_ACTION_SCALE
  # 前腿(arm)作为前脚要抬/落,缩放与 hip_pitch 同档(比双足 arm 放宽)。
  assert QUAD_ACTION_SCALE[r".*_arm_pitch_joint"] == 1.0
  assert QUAD_ACTION_SCALE[r".*_hip_pitch_joint"] == 1.0


def test_quad_sensors_present() -> None:
  """四足任务应配 feet_ground_contact(后脚承载)与 self_collision 传感器。"""
  cfg = load_env_cfg(QUAD_TASK)
  sensor_names = {s.name for s in (cfg.scene.sensors or ())}
  assert "feet_ground_contact" in sensor_names
  assert "self_collision" in sensor_names


# ---- 共享任务/常量未变断言 ----


def test_biped_upright_and_termination_unchanged() -> None:
  """双足任务仍用共享 upright(目标 +Z),四足替换函数未渗入共享。"""
  cfg = load_env_cfg(BIPED_TASK)
  upright = cfg.rewards["upright"]
  assert isinstance(upright, RewardTermCfg)
  # 双足 upright 仍是 velocity.mdp 里的共享类,不是 quad_upright。
  assert upright.func is not quad_upright
  assert upright.func.__module__ == "mjlab.tasks.velocity.mdp.rewards"

  fell = cfg.terminations["fell_over"]
  assert isinstance(fell, TerminationTermCfg)
  # 双足 fell_over 仍是共享 bad_orientation(不是四足 quad_bad_orientation)。
  assert fell.func is not quad_bad_orientation


def test_biped_action_scale_and_base_height_unchanged() -> None:
  """双足任务的 action scale 与 base_height target 保持不变。"""
  biped = load_env_cfg(BIPED_TASK)
  joint_pos = biped.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  assert isinstance(joint_pos.scale, dict)
  # 双足仍用自己那一档 action scale,不是四足 QUAD_ACTION_SCALE 覆盖的。
  assert joint_pos.scale != QUAD_ACTION_SCALE
  # 四足放宽了前腿 arm_pitch 到 0.5;双足不是这个值 → 四足覆盖未渗入双足。
  assert joint_pos.scale[r".*_arm_pitch_joint"] != 0.5
  # 双足 base_height 目标保持 0.17(人形高度),不被四足 0.07 覆盖。
  assert biped.rewards["base_height"].params["target_height"] == 0.17


def test_shared_constants_unpolluted() -> None:
  """共享 pikachu_constants 不应引入任何四足专属定义。"""
  for attr in ("QUAD_START", "QUAD_TILT_DEG", "QUAD_BODY_AXIS", "QUAD_ACTION_SCALE"):
    assert not hasattr(_shared_constants, attr), (
      f"shared pikachu_constants unexpectedly has {attr}"
    )
