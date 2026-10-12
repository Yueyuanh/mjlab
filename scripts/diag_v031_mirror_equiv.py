"""V031 左右腿镜像等价性的数值判定。

背景：``pikachu_v031_constants.py`` 的 docstring 声称左右轴约定是
「hip_pitch/hip_roll/hip_yaw/knee/ankle 反号，knee_up/knee_mid 同号」。
这个符号表是**镜像数据增强 / mirror loss 的唯一依据**，如果它是错的，
任何对称化训练都会注入错误约束。本脚本不靠读轴向量，而是直接做 FK 判定：

  对每一张候选符号表 s（关节 -> ±1），采样随机左腿驱动角 q_L，
  令 q_R = s * q_L，做正运动学，比较左右同名 body 的世界位置是否满足
  镜像 ``x_L == M @ x_R``（M = diag(1,-1,1)）。

  残差最小的那张表就是这台机器真实的镜像约定。若所有表残差都大，
  说明左右腿**根本不是镜像机构**（那就是资产问题）。

用法：
    uv run python scripts/diag_v031_mirror_equiv.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.pikachu.pikachu_v031_constants import PIKACHU_V031_XML

M = np.diag([1.0, -1.0, 1.0])

DRIVEN = ("hip_pitch", "hip_roll", "hip_yaw", "knee_up", "ankle")
ALL_JOINTS = DRIVEN + ("knee", "knee_mid")

TABLES: dict[str, dict[str, int]] = {
  "mixed (constants docstring)": {
    "hip_pitch": -1,
    "hip_roll": -1,
    "hip_yaw": -1,
    "knee_up": +1,
    "knee": -1,
    "knee_mid": +1,
    "ankle": -1,
  },
  "all_flip": dict.fromkeys(ALL_JOINTS, -1),
  "all_same": dict.fromkeys(ALL_JOINTS, +1),
  "knee_up+knee_mid flip only": {
    "hip_pitch": +1,
    "hip_roll": +1,
    "hip_yaw": +1,
    "knee_up": -1,
    "knee": +1,
    "knee_mid": -1,
    "ankle": +1,
  },
}


def main() -> None:
  m = mujoco.MjModel.from_xml_path(str(PIKACHU_V031_XML))
  d = mujoco.MjData(m)

  def qadr(joint: str, side: str) -> int:
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")
    assert j >= 0, f"缺关节 {side}_{joint}_joint"
    return m.jnt_qposadr[j]

  # 需要比较的 body：左右成对。
  pairs: list[tuple[str, int, int]] = []
  for suffix in (
    "hip_pitch_link",
    "hip_roll_link",
    "hip_yaw_link",
    "knee_link",
    "knee_up_link",
    "knee_mid_link",
    "ankle_link",
  ):
    lb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"left_{suffix}")
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"right_{suffix}")
    assert lb >= 0 and rb >= 0, f"缺 body {suffix}"
    pairs.append((suffix, lb, rb))

  free_qadr = m.jnt_qposadr[
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
  ]

  rng = np.random.default_rng(0)
  n_samples = 200
  # 所有关节都在各自限位内采样（±0.25 对 7 个关节都合法）。
  q_left = {j: rng.uniform(-0.25, 0.25, n_samples) for j in ALL_JOINTS}

  print(f"模型: {PIKACHU_V031_XML.name}")
  print(f"采样 {n_samples} 组随机左腿姿态，比较 {len(pairs)} 对 body 的镜像残差\n")
  print(f"{'候选符号表':<30}{'平均残差(mm)':>14}{'最大残差(mm)':>14}")
  print("-" * 60)

  results: dict[str, float] = {}
  for tname, signs in TABLES.items():
    worst = 0.0
    total = 0.0
    count = 0
    for s in range(n_samples):
      mujoco.mj_resetData(m, d)
      d.qpos[:] = 0.0
      d.qpos[free_qadr + 3] = 1.0  # free joint 四元数 w=1
      for j in ALL_JOINTS:
        d.qpos[qadr(j, "left")] = q_left[j][s]
        d.qpos[qadr(j, "right")] = signs[j] * q_left[j][s]
      mujoco.mj_forward(m, d)
      for _, lb, rb in pairs:
        res = np.linalg.norm(d.xpos[lb] - M @ d.xpos[rb])
        total += res
        count += 1
        worst = max(worst, res)
    mean_mm = total / count * 1000
    results[tname] = mean_mm
    print(f"{tname:<30}{mean_mm:>14.3f}{worst * 1000:>14.3f}")

  print("-" * 60)
  best = min(results, key=results.get)
  print(f"\n最小残差表: **{best}**  ({results[best]:.3f} mm)")
  if results[best] > 5.0:
    print("!! 所有候选表的残差都 > 5 mm —— 左右腿可能不是镜像机构，需要查资产。")
  else:
    print("该表即 V031 真实的左右镜像符号约定。")

  # 额外：只做「左右完全相同的角」这一平凡检查，看残差有多大 ——
  # 这是「策略若把左右当同号处理」会承受的几何误差。
  worst_same = 0.0
  for s in range(n_samples):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = 0.0
    d.qpos[free_qadr + 3] = 1.0
    for j in ALL_JOINTS:
      d.qpos[qadr(j, "left")] = q_left[j][s]
      d.qpos[qadr(j, "right")] = q_left[j][s]
    mujoco.mj_forward(m, d)
    for _, lb, rb in pairs:
      worst_same = max(worst_same, np.linalg.norm(d.xpos[lb] - M @ d.xpos[rb]))
  print(f"\n（对照）若左右取相同角，最大镜像残差 = {worst_same * 1000:.1f} mm")


if __name__ == "__main__":
  main()
