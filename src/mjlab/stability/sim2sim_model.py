"""27-dof 训练体的裸 MuJoCo 模型导入（sim2sim 的底座）。

旁路 mjlab 高层 env，只在 CPU Mujoco 上把 ``pikachu_27dof_sim2sim.xml`` 组装成
一张**带地面 + 带 position 执行器**的模型：

  - sim2sim 专用 XML 在训练体（不含地面）顶层恢复 ``<include file="scene.xml"/>``，
    由 scene.xml 提供地面（floor 已归零到 z=0）与光照/天空，cpu 部署即用；
  - 训练体 XML 不含执行器（由 ``PIKACHU_27DOF_ARTICULATION`` 编译期注入），
    这里逐关节注入与训练完全一致的 position 执行器（同一 Kp/Kd/effort/armature），
    使 ``data.ctrl`` 的布局与训练对齐、由 MuJoCo position 执行器天然做 PD。

本模块只做**模型导入**：不接策略、不做观测、不控制关节。验证时把关节固定到
``KNEES_BENT_27DOF_KEYFRAME`` 起步姿态，在 viewer 里确认模型导入无误。
``data.ctrl`` 保持起步位置即"固定关节"。

后续：``policy.py`` / ``obs.py`` / ``main.py``（v025 式控制循环）在此之上扩展。
"""

from __future__ import annotations

import re

import mujoco
import mujoco.viewer

from mjlab.actuator import BuiltinPositionActuatorCfg  # 仅类型提示
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_constants import (
  ALL_JOINT_NAMES,
  KNEES_BENT_27DOF_KEYFRAME,
  PIKACHU_27DOF_ARTICULATION,
  PIKACHU_27DOF_SIM2SIM_XML,
)
from mjlab.utils.spec import create_position_actuator

# 起步姿态（README 复用）：固定关节 = 把 ctrl 保持为这些目标角。
_KF_JOINT_POS = KNEES_BENT_27DOF_KEYFRAME.joint_pos
assert _KF_JOINT_POS is not None
KNEES_BENT_START = _KF_JOINT_POS


def _actuator_entries() -> dict[str, tuple[float, float, float | None, float | None]]:
  """按关节名收集执行器参数 (stiffness, damping, effort_limit, armature)。

  ``PIKACHU_27DOF_ARTICULATION.actuators`` 是三个 ``BuiltinPositionActuatorCfg``，
  各自带一组 ``target_names_expr`` 正则；对 ``ALL_JOINT_NAMES`` 里每个关节，
  用命中它的第一个配置取参，保证与训练体 ctrl 顺序一致（关节定义顺序）。
  """
  params_by_expr: list[tuple[tuple[str, ...], BuiltinPositionActuatorCfg]] = []
  for cfg in PIKACHU_27DOF_ARTICULATION.actuators:
    assert isinstance(cfg, BuiltinPositionActuatorCfg)
    params_by_expr.append((tuple(cfg.target_names_expr), cfg))

  entries: dict[str, tuple[float, float, float | None, float | None]] = {}
  for name in ALL_JOINT_NAMES:
    for exprs, cfg in params_by_expr:
      if any(re.fullmatch(p, name) for p in exprs):
        entries[name] = (cfg.stiffness, cfg.damping, cfg.effort_limit, cfg.armature)
        break
    else:
      raise ValueError(
        f"Joint {name!r} matched no actuator in PIKACHU_27DOF_ARTICULATION."
      )
  return entries


def build_model() -> mujoco.MjModel:
  """编译 27-dof sim2sim 体（带地面 + 带 position 执行器），返回 CPU Mujoco 模型。

  - 复用 ``PIKACHU_27DOF_SIM2SIM_XML``（训练体 + scene.xml，地面由 scene 提供）；
  - 执行器：按 ``ALL_JOINT_NAMES`` 顺序逐个 ``create_position_actuator``。
  """
  spec = mujoco.MjSpec.from_file(str(PIKACHU_27DOF_SIM2SIM_XML))

  # 注入 position 执行器（顺序 = ALL_JOINT_NAMES，与训练 d.ctrl 对齐）。
  for name, (stiffness, damping, effort_limit, armature) in _actuator_entries().items():
    create_position_actuator(
      spec,
      name,
      stiffness=stiffness,
      damping=damping,
      effort_limit=effort_limit,
      armature=armature,
    )

  return spec.compile()


def apply_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """把起步姿态 ``KNEES_BENT_27DOF_KEYFRAME`` 写入 ``data``（自由根 + 27 关节）。

  关节按 ``ALL_JOINT_NAMES`` 定位到 ``qpos[7:]``（与 XML 定义顺序一致），
  根 pos 写 ``qpos[0:3]``、根 quat（wxyz，恒等）写 ``qpos[3:7]``；随后
  ``mj_forward`` 使姿态/接触成立。
  """
  mujoco.mj_resetData(model, data)
  keyframe = KNEES_BENT_27DOF_KEYFRAME
  joint_pos = keyframe.joint_pos
  assert joint_pos is not None

  data.qpos[:3] = keyframe.pos
  data.qpos[3:7] = keyframe.rot if keyframe.rot is not None else (1.0, 0.0, 0.0, 0.0)
  for i, name in enumerate(ALL_JOINT_NAMES):
    data.qpos[7 + i] = joint_pos.get(name, 0.0)
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def _freeze_targets() -> list[float]:
  """固定关节的目标角：起步姿态里全部 27 关节角（position 执行器绝对目标）。"""
  return [KNEES_BENT_START.get(name, 0.0) for name in ALL_JOINT_NAMES]


def main() -> None:
  """验证模型导入：编译 → 起步姿态 → viewer 中固定关节稳定站立。"""
  model = build_model()
  data = mujoco.MjData(model)

  apply_keyframe(model, data)
  targets = _freeze_targets()
  data.ctrl[:] = targets

  n_ground = sum(
    1
    for i in range(model.ngeom)
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) == "floor"
  )
  print(
    f"[sim2sim_model] 关节数={model.nq - 7} 执行器数(nu)={model.nu} "
    f"scene地面geom={n_ground} ctrl长度={data.ctrl.shape[0]}"
  )

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      # 固定关节：position 执行器把每个关节锁在起步目标角。
      data.ctrl[:] = targets
      mujoco.mj_step(model, data)
      viewer.sync()


if __name__ == "__main__":
  main()
