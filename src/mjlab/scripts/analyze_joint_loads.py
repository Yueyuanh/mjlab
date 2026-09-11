"""基于运动学 + 动力学推算各关节静态扭矩与动态加速扭矩（固定 base_link）。

**固定机架**：把浮动基座锁在默认站立位姿（qpos 前 7 维不动），只变关节角。

**静态力（扭矩支撑的力，力×力臂）**
  - 每个关节承受的**静态力**＝其远端子树全部质量 × g（仅由关节所在 body 的拓扑
    决定，与姿态无关，单位 N）。
  - 每个关节的**静态力矩**＝克服远端重力负载所需的平衡扭矩＝``data.qfrc_bias``
    （qvel=0 时即重力项，含科氏为 0）。它就是「力 × 力臂」中的力矩。
  - **力臂**＝扭矩 / 静态力（远端负载对关节轴的等效力臂，单位 m）。

**动态加速扭矩**
  - ``mj_forward`` 后取稀疏质量阵 ``data.qM`` → ``mj_fullM`` 组装满阵 M。
  - 每关节**有效惯量** M_ii（绕自身自由度的惯性项，单位 kg·m²）；对名义角加速度
    ``a0`` 的动态加速扭矩 ≈ ``M_ii * a0``（固定 base、其余关节不动时）。

**扫描取极值**：每个关节独立在其限位范围 ``jnt_range`` 内扫角（其余关节保持默认
站姿），记录每关节自身 ``qfrc_bias`` 的**最大绝对值**作为「最大静态力矩」。

输出：逐关节数值清单（stdout 打印）+ matplotlib 图像（--output，默认 /tmp）。

用法（mjlab 目录下）：

  uv run python -m mjlab.scripts.analyze_joint_loads \
      --output /tmp/joint_loads.png --save-csv /tmp/joint_loads.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
import mujoco  # noqa: E402

from mjlab.mocap.analyze_stability import DEFAULT_XML  # noqa: E402

NOMINAL_ACCEL = 5.0  # 动态扭矩标称角加速度 rad/s²
SWEEP_STEPS = 41  # 每关节扫角点数


def _entity_name(model: mujoco.MjModel, adr_array: np.ndarray, i: int) -> str:
  """从 names 缓冲按 *_adr 取第 i 个实体名（'' 表示无名）。"""
  adr = int(adr_array[i])
  if adr < 0:
    return ""
  end = model.names.find(b"\x00", adr)
  if end < 0:
    end = len(model.names)
  try:
    return model.names[adr:end].decode()
  except UnicodeDecodeError:
    return ""


def _distal_mass_factors(
  model: mujoco.MjModel,
) -> tuple[list[int], list[float]]:
  """返回 (hinge 关节 id 列表, 每关节远端子树总质量 kg)。

  用 body 父子链（body_parentid）递归求每个 body 的子树质量；关节所属 body 的
  子树质量即该关节承受的重力负载质量。
  """
  nb = model.nbody
  parent = np.asarray(model.body_parentid, dtype=int)
  children: list[list[int]] = [[] for _ in range(nb)]
  for b in range(1, nb):
    children[int(parent[b])].append(b)
  mass = np.asarray(model.body_mass, dtype=float)

  def subtree(root: int) -> float:
    return mass[root] + sum(subtree(c) for c in children[root])

  hinge_ids = [
    j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  subtree_mass = [subtree(int(model.jnt_bodyid[j])) for j in hinge_ids]
  return hinge_ids, subtree_mass


def analyze(model: mujoco.MjModel, n_steps: int = SWEEP_STEPS) -> dict:
  """核心：固定 base，扫姿态算每关节静态力/力矩/惯量/动态扭矩。"""
  data = mujoco.MjData(model)
  ngc = model.nq - 7  # 27 hinge 关节
  hinge_ids, subtree_mass = _distal_mass_factors(model)
  joint_dof = [int(model.jnt_dofadr[j]) for j in hinge_ids]
  names = [_entity_name(model, model.name_jntadr, j) for j in hinge_ids]
  ranges = np.asarray(
    [(model.jnt_range[j][0], model.jnt_range[j][1]) for j in hinge_ids]
  )
  qpos_slot = 7 + np.arange(ngc)  # dof - 6，因全为 hinge 且顺序一致
  g = 9.81

  # 每关节远端子树质量 → 静态力（N，姿态无关）。
  static_force = np.array(subtree_mass) * g
  # 等效力臂（默认站姿）。
  # --- 默认站姿静态力矩（qfrc_bias）---
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  default_static_torque = np.array([float(data.qfrc_bias[d]) for d in joint_dof])

  # 质量阵 → 每关节有效惯量 M_ii 与动态加速扭矩。
  # mj_fullM(model, data, dst) 从 data 内部稀疏质量阵展开成稠密 nv×nv。
  M_full = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, M_full)
  M = np.asarray(M_full)
  eff_inertia = np.array([M[d, d] for d in joint_dof])
  dyn_torque = eff_inertia * NOMINAL_ACCEL
  # 四元数里 M_ii 读的是默认站姿处的构型惯量，扫姿态时另算（见下）。

  # --- 逐关节扫角，取每个关节自身 |qfrc_bias| 的最大值 ---
  max_static_torque = np.zeros(ngc)
  torque_curves = np.zeros((ngc, n_steps))
  sweep_angles = np.zeros((ngc, n_steps))
  for i in range(ngc):
    lo, hi = ranges[i]
    th = np.linspace(lo, hi, n_steps)
    sweep_angles[i] = th
    qj = np.array(model.qpos0)
    for k, a in enumerate(th):
      qj[qpos_slot[i]] = a
      data.qpos[:] = qj
      data.qvel[:] = 0.0
      mujoco.mj_forward(model, data)
      torque_curves[i, k] = float(data.qfrc_bias[joint_dof[i]])
    max_static_torque[i] = np.abs(torque_curves[i]).max()

  # 等效力臂：最大静态力矩处对应默认站姿参考（用默认值给出数量级即可）。
  with np.errstate(divide="ignore", invalid="ignore"):
    lever_default = np.divide(
      np.abs(default_static_torque),
      static_force,
      out=np.zeros(ngc),
      where=static_force > 0,
    )
    lever_max = np.divide(
      max_static_torque, static_force, out=np.zeros(ngc), where=static_force > 0
    )

  return {
    "names": names,
    "joint_dof": joint_dof,
    "static_force_N": static_force,
    "default_static_torque_Nm": default_static_torque,
    "max_static_torque_Nm": max_static_torque,
    "lever_default_m": lever_default,
    "lever_max_m": lever_max,
    "eff_inertia": eff_inertia,
    "dyn_torque_Nm": dyn_torque,
    "torque_curves": torque_curves,
    "sweep_angles": sweep_angles,
    "ranges": ranges,
    "qpos_slot": qpos_slot,
    "model": model,
  }


def _plot(res: dict, output: Path) -> None:
  """画四联图：最大静态力矩 / 静态力 / 有效惯量与动态扭矩 / 扭矩-转角曲线。"""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  names = res["names"]
  x = np.arange(len(names))
  fig, ax = plt.subplots(2, 2, figsize=(16, 9))
  fig.suptitle("Per-joint static & dynamic loads (base fixed, 27 dof)", fontsize=14)

  # (a) 最大静态力矩
  tmax = res["max_static_torque_Nm"]
  ax[0, 0].barh(x, tmax, color="#c0392b")
  ax[0, 0].set_yticks(x, names)
  ax[0, 0].invert_yaxis()
  for i, v in enumerate(tmax):
    ax[0, 0].text(v, i, f" {v:.2f}", va="center", fontsize=8)
  ax[0, 0].set_title("Max static torque (gravity, force x lever) [N m]")
  ax[0, 0].set_xlabel("N m")

  # (b) 静态支撑力（子树重量）
  f = res["static_force_N"]
  ax[0, 1].barh(x, f, color="#2980b9")
  ax[0, 1].set_yticks(x, names)
  ax[0, 1].invert_yaxis()
  for i, v in enumerate(f):
    ax[0, 1].text(v, i, f" {v:.1f}", va="center", fontsize=8)
  ax[0, 1].set_title("Static support force (distal subtree weight) [N]")
  ax[0, 1].set_xlabel("N")

  # (c) 有效惯量与动态加速扭矩
  eff = res["eff_inertia"]
  adyn = res["dyn_torque_Nm"]
  ax[1, 0].barh(x, eff, color="#16a085", label="effective inertia M_ii [kg m^2]")
  ax[1, 0].set_yticks(x, names)
  ax[1, 0].invert_yaxis()
  for i, v in enumerate(eff):
    ax[1, 0].text(v, i, f" {v:.3f}", va="center", fontsize=8)
  ax[1, 0].set_title(
    f"Dynamic accel torque at {NOMINAL_ACCEL:.0f} rad/s^2 [N m] (inertia)"
  )
  ax[1, 0].set_xlabel("N m")
  axin = ax[1, 0].twiny()
  axin.barh(x, adyn, color="#e67e22", alpha=0.0)
  for i, v in enumerate(adyn):
    axin.text(v, i, f"{v:.2f}", ha="right", va="center", fontsize=7, color="#e67e22")

  # (d) 扭矩-转角曲线（挑 5 个最大静态力矩的关节）
  order = np.argsort(tmax)[::-1][:5]
  curves = res["torque_curves"]
  angles = res["sweep_angles"]
  for i in order:
    ax[1, 1].plot(angles[i], curves[i], label=names[i], lw=2)
  ax[1, 1].set_title("Static torque vs joint sweep angle")
  ax[1, 1].set_xlabel("joint angle [rad]")
  ax[1, 1].set_ylabel("qfrc_bias [N m]")
  ax[1, 1].legend(fontsize=8, loc="best")

  fig.tight_layout()
  fig.savefig(output, dpi=150)
  print(f"已保存图像: {output.resolve()}")


def _print_table(res: dict, save_csv: Path | None) -> None:
  """打印逐关节数值清单（stdout + 可选 CSV）。"""
  n = len(res["names"])
  header = (
    f"{'joint':<22}{'static_F(N)':>12}{'T_default(Nm)':>14}"
    f"{'T_max(Nm)':>12}{'lever(m)':>10}{'Mii(kgm2)':>11}{'T_dyn(Nm)':>10}"
  )
  print(header)
  print("-" * len(header))
  rows = []
  for i in range(n):
    rows.append(
      (
        res["names"][i],
        res["static_force_N"][i],
        res["default_static_torque_Nm"][i],
        res["max_static_torque_Nm"][i],
        res["lever_max_m"][i],
        res["eff_inertia"][i],
        res["dyn_torque_Nm"][i],
      )
    )
    print(
      f"{rows[-1][0]:<22}{rows[-1][1]:>12.1f}{rows[-1][2]:>14.3f}"
      f"{rows[-1][3]:>12.3f}{rows[-1][4]:>10.3f}{rows[-1][5]:>11.4f}"
      f"{rows[-1][6]:>10.3f}"
    )
  print(f"\nNOMINAL_ACCEL = {NOMINAL_ACCEL:.0f} rad/s^2（固定 base、其余关节不动）")
  print("static_F = 该关节远端子树总重 (N)；T_max = 独立扫角下 |qfrc_bias| 最大值；")
  print("lever = T/static_force（远端负载等效力臂）；T_dyn = M_ii * NOMINAL_ACCEL。")
  if save_csv is not None:
    np.savetxt(
      save_csv,
      np.array([r[1:] for r in rows]),
      delimiter=",",
      header=",".join(
        ["static_force_N", "T_default_Nm", "T_max_Nm", "lever_m", "Mii", "T_dyn_Nm"]
      ),
      comments="",
      fmt="%.6f",
    )
    print(f"已导出 CSV: {save_csv.resolve()}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
  ap.add_argument("--output", type=Path, default=Path("/tmp/joint_loads.png"))
  ap.add_argument("--save-csv", type=Path, default=None)
  ap.add_argument("--accel", type=float, default=NOMINAL_ACCEL, help="动态标称角加速度")
  ap.add_argument("--steps", type=int, default=SWEEP_STEPS, help="每关节扫角点数")
  args = ap.parse_args()

  model = mujoco.MjModel.from_xml_path(str(args.xml))
  print(f"输入模型: {args.xml.resolve()}  总质量: {model.body_mass.sum():.3f} kg")
  print(f"固定 base_link，动力学校准 @ g=9.81，accel={args.accel:.1f} rad/s^2\n")

  res = analyze(model, n_steps=args.steps)
  res["dyn_torque_Nm"] = res["eff_inertia"] * args.accel
  _print_table(res, args.save_csv)
  _plot(res, args.output)


if __name__ == "__main__":
  main()
