"""离线/在线根据运动学 + 各关节惯量计算整机 CoM 与 ZMP。

复用 :mod:`mjlab.mocap.stability` 的合成逻辑（与 ``analyze_stability.py`` 同一套
平行轴口径）：整机质心 CoM、绕其惯量张量及主惯量，再用线性倒立摆近似由 CoM 轨迹
差分出水平 ZMP。

**离线模式**（``--motion <file.npz>``）：
    读一条 tracking/录制好的 motion npz（keys 含 ``body_pos_w``/``body_quat_w``，
    50fps，见 ``scripts/csv_to_npz.py``），逐帧把每条 body 的世界原点位姿当
    ``xpos/xquat`` 喂给 ``inertia_about_com``，得到逐帧 CoM/惯量，再用整个 CoM
    轨迹算 ZMP。导出 CSV（``--output``）供离线分析。纯 CPU host MjModel，秒出。

**单姿态模式**（``--pose "q0 q1 ..."``，可选）：
    单次评估一个关节角姿态的 CoM/惯量与静态稳定裕度。``--pose`` 给的是除浮动基座
    外、按模型关节顺序（qpos 第 7 位起）的关节角（rad）；基座用默认站立姿态。

**在线**：``play --com`` 已把同一套计算实时叠加到 native/viser 查看器
（:func:`mjlab.mocap.stability.com_and_inertia_from_sim`）。

用法（mjlab 目录下）：

  uv run python -m mjlab.scripts.analyze_com_zmp \
      --motion src/mjlab/mocap/npz/Pikachu-Transition-Down.npz \
      --output /tmp/com_zmp.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from mjlab.mocap.analyze_stability import DEFAULT_XML
from mjlab.mocap.stability import _host_cinert, inertia_about_com, zmp_from_com

# 每 motion 默认 50fps（与 csv_to_npz 输出一致），可用 npz 里 fps 字段覆盖。
DEFAULT_FPS = 50.0


def _load_model(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
  """从 XML 加载宿主模型与数据（纯只读，不出图）。"""
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  return model, data


def _inertia_from_poses(
  mass, cinert, ipos, iquat, xpos, xquat
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """逐帧求 (com, inertia, principal_moments, principal_axes)。

  Args:
    mass/cinert/ipos/iquat: 宿主模型静态量（nbody 维度）。
    xpos: (T, nbody, 3) 或 (nbody, 3)。
    xquat: (T, nbody, 4) 或 (nbody, 4)，wxyz。
  """
  xpos = np.asarray(xpos, dtype=np.float64)
  xquat = np.asarray(xquat, dtype=np.float64)
  single = xpos.ndim == 2
  if single:
    xpos = xpos[None, ...]
    xquat = xquat[None, ...]

  coms = np.zeros((xpos.shape[0], 3))
  inertias = np.zeros((xpos.shape[0], 3, 3))
  moments = np.zeros((xpos.shape[0], 3))
  axes = np.zeros((xpos.shape[0], 3, 3))
  for t in range(xpos.shape[0]):
    com, inertia = inertia_about_com(mass, cinert, ipos, iquat, xpos[t], xquat[t])
    eigh, eigvec = np.linalg.eigh(inertia)
    order = np.argsort(eigh)[::-1]
    coms[t] = com
    inertias[t] = inertia
    moments[t] = eigh[order]
    axes[t] = eigvec[:, order].T
  return (
    (coms, inertias, moments, axes)
    if not single
    else (coms[0], inertias[0], moments[0], axes[0])
  )


def _make_qpos(
  model: mujoco.MjModel, data: mujoco.MjData, pose: np.ndarray
) -> np.ndarray:
  """用给定关节角（模型关节序）生成完整 qpos。

  以模型的默认 qpos（``qpos0``，含浮动基座位姿）为基，覆盖第 7 位起的关节角；
  需调用方 ``mj_forward`` 后从 ``data.xpos/xquat`` 读真实 body 位姿。
  """
  del data
  qpos = np.asarray(pose, dtype=np.float64)
  if qpos.size != model.nq - 7:
    raise ValueError(
      f"--pose 应给 {model.nq - 7} 个关节角（模型关节序，rad），实际 {qpos.size}"
    )
  full = np.array(model.qpos0)
  full[7:] = qpos
  return full


def _static_margin(com: np.ndarray, foot_half_width: float) -> float:
  """静态侧向稳定裕度：CoM 地面投影到支撑外缘的最近距离（近似正视面矩形）。"""
  return foot_half_width - abs(com[1])


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
  ap.add_argument(
    "--motion",
    type=Path,
    default=None,
    help="离线读取的 motion npz（含 body_pos_w / body_quat_w）。",
  )
  ap.add_argument(
    "--pose",
    type=str,
    default=None,
    help='单姿态关节角（模型关节序，rad，空格分隔，如 "0 0 ... 0"）。',
  )
  ap.add_argument(
    "--fps", type=float, default=DEFAULT_FPS, help="motion 帧率（默认 %(default)s）。"
  )
  ap.add_argument(
    "--output",
    type=Path,
    default=None,
    help="导出逐帧 CoM/惯量/ZMP 的 CSV 路径（仅 --motion 模式）。",
  )
  args = ap.parse_args()

  model, data = _load_model(args.xml)
  mass = model.body_mass
  cinert = _host_cinert(model)
  ipos = model.body_ipos
  iquat = model.body_iquat
  print(f"输入模型: {args.xml}  总质量: {mass.sum():.3f} kg\n")

  if args.motion is not None:
    dz = np.load(args.motion)
    xpos = dz["body_pos_w"]
    xquat = dz["body_quat_w"]
    fps = float(dz.get("fps", [args.fps])[0]) if "fps" in dz else args.fps
    dt = 1.0 / fps
    T = xpos.shape[0]
    nbody = len(mass)
    # npz 若缺世界 body（index 0），前后补一条零位姿的 world 身体对齐索引。
    offset = 0 if xpos.shape[1] == nbody else 1
    if offset:
      print(
        f"[INFO] motion 有 {xpos.shape[1]} 个 body，模型 {nbody} 个，补 world 对齐。"
      )
    X = np.zeros((T, nbody, 3))
    Q = np.zeros((T, nbody, 4))
    Q[:, :, 0] = 1.0  # wxyz 单位四元数
    X[:, offset:, :] = xpos[:, : nbody - offset]
    Q[:, offset:, :] = xquat[:, : nbody - offset]
    coms, inertias, moments, _ = _inertia_from_poses(mass, cinert, ipos, iquat, X, Q)
    zmp = zmp_from_com(coms, dt, g=9.81)
    times = np.arange(T) * dt

    if args.output is not None:
      out = np.column_stack(
        (
          np.arange(T),
          times,
          coms,
          moments,
          zmp,
        )
      )
      header = "frame,time_s,com_x,com_y,com_z,I_xx,I_yy,I_zz,zmp_x,zmp_y"
      np.savetxt(args.output, out, delimiter=",", header=header, comments="")
      print(f"已导出 {T} 帧到: {args.output.resolve()}")

    # 概览。
    print(
      f"离线 {T} 帧 @ {fps:.0f}fps (dt={dt:.4f}s, {T * dt:.2f}s)：\n"
      f"  CoM z 范围: [{coms[:, 2].min():.3f}, {coms[:, 2].max():.3f}] m "
      f"(均值 {coms[:, 2].mean():.3f})\n"
      f"  CoM x 范围: [{coms[:, 0].min():.3f}, {coms[:, 0].max():.3f}] m\n"
      f"  主惯量 Izz 范围: [{moments[:, 2].min():.4f}, {moments[:, 2].max():.4f}] kg·m²\n"
      f"  ZMP x 范围: [{zmp[:, 0].min():.3f}, {zmp[:, 0].max():.3f}] m\n"
      f"  ZMP y 范围: [{zmp[:, 1].min():.3f}, {zmp[:, 1].max():.3f}] m"
    )
    return

  if args.pose is not None:
    qpos = _make_qpos(model, data, np.array([float(x) for x in args.pose.split()]))
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    xpos = np.asarray(data.xpos, dtype=np.float64)
    xquat = np.asarray(data.xquat, dtype=np.float64)
    com, _inertia, moments, axes = _inertia_from_poses(
      mass, cinert, ipos, iquat, xpos, xquat
    )
    print(
      f"单姿态：\n  CoM = ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f}) m\n"
      f"  主惯量 I = ({moments[0]:.4f}, {moments[1]:.4f}, {moments[2]:.4f}) kg·m²\n"
      f"  主轴:\n"
      + "\n".join(f"    ({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})" for a in axes)
    )
    return

  ap.error("必须提供 --motion 或 --pose 之一。")


if __name__ == "__main__":
  main()
