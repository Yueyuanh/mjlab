"""RMG_AMP(硬件格式 CSV) → tracking npz 批量转换。

与 ``csv_to_npz_pikachu.py`` 共用同一套 sim/FK 管线与 npz 字段,差异只在**输入
CSV 的列布局**:

  - ``csv_to_npz_pikachu.py`` 吃 ``mjlab.mocap`` 录制的 21 列 CSV
    (root_pos 3 + root_quat xyzw 4 + dof 14,无表头)。
  - 本脚本吃 rmg 的 ``gait_generator.py --hardware`` 输出:47 列 **带表头**
    (root_pos 3 + root_quat 4 + 腿关节 10 + toe 6 + world 线/角速度 6 +
    joint_vel 10 + toe_vel 6 + foot_contact 2)。

三个必须记住的格式坑(全部在下面显式处理):

1. **表头骗人**:表头写 ``root_quat_w, root_quat_x, ...``,但数据实际是
   ``xyzw`` 序(与 ``json_to_txt.py`` 写出的 ``QuaternionOrder: "xyzw"`` 一致)。
   读取后按 xyzw 解释再转 wxyz 交给 MuJoCo。
2. **只有 10 个腿关节**:手臂 4 关节不在参考里。写 sim 时只写这 10 个
   (``LEG_JOINT_NAMES`` 序),手臂保持 ``default_joint_pos``,因此 FK 出来的
   ``body_*_w`` 是"腿在动、手臂归位"的合理姿态。
3. **表头行**:``skiprows=1``,否则 ``np.loadtxt`` 会因非数值列报错。

输出 = ``mocap/npz/pikachu_rmgamp_npz/<clip>.npz``,字段与 tracking 的
``MotionLoader`` 完全对齐(``fps/joint_pos/joint_vel/body_*_w``)。

用法::

    uv run python -m mjlab.scripts.csv_to_npz_rmgamp \
        --input-dir src/mjlab/mocap/csv/RMG_AMP/csv \
        --output-dir src/mjlab/mocap/npz/pikachu_rmgamp_npz \
        --device cuda:0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import LEG_JOINT_NAMES
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_lite import (
  get_pikachu_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.mdp.multi_motion import parse_clip_command
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)

# RMG_AMP 硬件 CSV 的列切片(0-based,含表头行已跳过)。
_COL_ROOT_POS = slice(0, 3)
_COL_ROOT_QUAT_XYZW = slice(3, 7)
_COL_LEG_DOF = slice(7, 17)  # 10 个腿关节,顺序 = LEG_JOINT_NAMES

# 命令标签的稳态窗口。rmg 每条 clip 的前 ~100 帧是起步/整定段(速度从 0 爬升),
# 用它算平均速度会把标签拉低一大截;取 150 帧之后才是真正的稳态步态。
STEADY_START_FRAME = 150


class RmgAmpMotionLoader:
  """读 RMG_AMP 硬件 CSV,插值到 ``output_fps`` 并差分出线/角速度。

  接口与 ``csv_to_npz_pikachu.MotionLoader`` 的 ``get_next_state`` 一致,方便
  复用同一套 sim 循环。
  """

  def __init__(
    self,
    motion_file: str,
    input_fps: float,
    output_fps: float,
    device: torch.device | str,
  ) -> None:
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / input_fps
    self.output_dt = 1.0 / output_fps
    self.current_idx = 0
    self.device = device
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self) -> None:
    raw = np.loadtxt(self.motion_file, delimiter=",", skiprows=1)
    motion = torch.from_numpy(raw).to(torch.float32).to(self.device)
    if motion.shape[1] < 17:
      raise ValueError(
        f"{self.motion_file}: expected >=17 columns (hardware format), "
        f"got {motion.shape[1]}"
      )
    self.motion_base_poss_input = motion[:, _COL_ROOT_POS]
    # 表头写的是 wxyz,数据实为 xyzw —— 按 xyzw 读入后重排成 wxyz。
    quat_xyzw = motion[:, _COL_ROOT_QUAT_XYZW]
    self.motion_base_rots_input = quat_xyzw[:, [3, 0, 1, 2]]
    self.motion_dof_poss_input = motion[:, _COL_LEG_DOF]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self) -> None:
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
      out[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return out

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self) -> None:
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  def get_next_state(self):
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = self.current_idx >= self.output_frames
    return state, reset_flag


def left_leg_sign(robot: Entity, leg_joint_ids: list[int]) -> torch.Tensor:
  """左腿各关节的符号修正 ``(10,)``,把 rmg 参考约定换算到 MJCF 约定。

  **为什么需要它**:rmg 的 URDF 是**左右对称**的(``left_knee_joint`` 与
  ``right_knee_joint`` 同为 ``axis="0 1 0"``、``range=[0, 1.57]``),而 mjlab 的
  MJCF(6 个 pikachu 变体一致,也是 velocity/transition 与部署所用的模型)
  **左腿三关节轴是镜像的**:``left_knee_joint axis="0 -1 0" range=[-1.57, 0]``。
  直接把 rmg 的关节角写进 MJCF,左膝会被限位钉死在 0,左腿整条不可达。

  实测(``probe_leg_sign.py``,clip ``0.185_0.0_0.0``):不修正时左腿最大越界
  **2.076 rad** 而右腿只有 0.504;按本函数修正后左 0.506 / 右 0.504 —— 左右
  完全对称,残余的 0.50 rad 是两条腿共有的 rmg 膝超限(119° vs 90° 限位),
  与符号无关。

  判据用**关节限位**:镜像关节的限位恰好互为相反数,即
  ``left_lo == -right_hi`` 且 ``left_hi == -right_lo``。本机器人上
  ``left_hip_pitch / left_knee / left_ankle`` 命中;``hip_roll``(左右同为
  ``[-0.09, 0.26]``)与 ``hip_yaw``(同)不命中。这与**关节轴点积**判据结论
  完全一致(镜像的三关节左右轴点积为 -1,另两个为 +1)。

  不硬编码关节名,换机器人时这里自动跟上:新模型若没有镜像,自然返回全 1。
  """
  lim = robot.data.joint_pos_limits[0]
  sign = torch.ones(len(LEG_JOINT_NAMES), device=lim.device)
  tol = 1e-6
  for i, name in enumerate(LEG_JOINT_NAMES):
    if not name.startswith("left_"):
      continue
    right = "right_" + name[len("left_") :]
    if right not in LEG_JOINT_NAMES:
      continue
    li = leg_joint_ids[i]
    ri = leg_joint_ids[LEG_JOINT_NAMES.index(right)]
    lo_l, hi_l = float(lim[li, 0]), float(lim[li, 1])
    lo_r, hi_r = float(lim[ri, 0]), float(lim[ri, 1])
    mirrored = abs(lo_l + hi_r) < tol and abs(hi_l + lo_r) < tol
    if mirrored and abs(lo_l - lo_r) > tol:
      sign[i] = -1.0
  return sign


def convert_one(
  csv_path: Path,
  output_path: Path,
  *,
  scene: Scene,
  sim: Simulation,
  robot: Entity,
  leg_joint_ids: list[int],
  input_fps: float,
  output_fps: float,
  device: str,
) -> dict[str, Any]:
  """跑一遍 FK,把一条 clip 写成 npz。返回该 clip 的统计信息。"""
  motion = RmgAmpMotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )
  sign = left_leg_sign(robot, leg_joint_ids)

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  scene.reset()
  vel_err_lin = 0.0
  vel_err_ang = 0.0
  frames = 0
  while True:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    # 只写 10 个腿关节;手臂留在 default pose(参考里没有手臂)。
    # ``* sign`` 把 rmg 的对称约定换算到 MJCF 的镜像左腿约定(见 ``left_leg_sign``);
    # 不换算的话左膝会被限位钉死,FK 出来的左腿 body 位姿也是错的。
    joint_pos[:, leg_joint_ids] = motion_dof_pos * sign
    joint_vel[:, leg_joint_ids] = motion_dof_vel * sign
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )

    # FK 速度 vs 差分速度的一致性校验(base body = 0 号)。
    vel_err_lin = max(
      vel_err_lin,
      float(
        torch.linalg.norm(robot.data.body_link_lin_vel_w[0, 0] - motion_base_lin_vel[0])
      ),
    )
    vel_err_ang = max(
      vel_err_ang,
      float(
        torch.linalg.norm(robot.data.body_link_ang_vel_w[0, 0] - motion_base_ang_vel[0])
      ),
    )
    frames += 1
    if reset_flag:
      break

  for k in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[k] = np.stack(log[k], axis=0)

  # 裁到 10 个腿关节。tracking 的 ``MotionCommand`` 要求参考的关节维数与策略
  # 动作空间一致(``joint_names=LEG_JOINT_NAMES`` -> ``_joint_ids`` 长度 10),
  # 否则 ``_write_reference_state_to_sim`` 会把 14 维参考写进 10 个关节而报错。
  # 手臂在参考里恒为 default pose、不参与跟踪,直接丢掉。
  n_legs = len(LEG_JOINT_NAMES)
  log["joint_pos"] = log["joint_pos"][:, :n_legs]
  log["joint_vel"] = log["joint_vel"][:, :n_legs]

  # 命令标签 = **实测稳态速度**,不是 rmg 的 dx 换算值。二者差距很大:rmg 的
  # ``pikachu_walk_0.37_0.0_0.0`` 档实际只走出 0.19 m/s(且与 0.185 档逐帧几乎
  # 相同),``0.3/±0.2`` 档只走出 (0.11, ∓0.07)。用文件名当命令会让"发 0.37 却
  # 只走 0.19",下游速度控制失去物理意义。这里按稳态段(见 STEADY_START_FRAME)
  # 的 base 线速度/yaw 角速度均值给出每条 clip 真实对应的命令。
  steady = slice(STEADY_START_FRAME, frames)
  body_lin_vel = np.gradient(log["body_pos_w"][:, 0, :], axis=0) * output_fps
  command = np.array(
    [
      body_lin_vel[steady, 0].mean(),
      body_lin_vel[steady, 1].mean(),
      log["body_ang_vel_w"][steady, 0, 2].mean(),
    ],
    dtype=np.float32,
  )
  log["command"] = command
  log["nominal_command"] = np.array(parse_clip_command(csv_path.stem), dtype=np.float32)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **log)

  body_pos = log["body_pos_w"]
  return {
    "clip": csv_path.stem,
    "frames": frames,
    "fps": output_fps,
    "fk_lin_vel_max_err": vel_err_lin,
    "fk_ang_vel_max_err": vel_err_ang,
    "travel_x": float(body_pos[-1, 0, 0] - body_pos[0, 0, 0]),
    "travel_y": float(body_pos[-1, 0, 1] - body_pos[0, 0, 1]),
    "mean_lin_vel": body_lin_vel.mean(axis=0).tolist(),
    "command": command.tolist(),
    "nominal_command": log["nominal_command"].tolist(),
    "nan": bool(np.isnan(log["joint_pos"]).any() or np.isnan(body_pos).any()),
  }


def main(
  input_dir: str = str(MJLAB_SRC_PATH / "mocap" / "csv" / "RMG_AMP" / "csv"),
  output_dir: str = str(MJLAB_SRC_PATH / "mocap" / "npz" / "pikachu_rmgamp_npz"),
  input_fps: float = 50.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
) -> None:
  """批量把 RMG_AMP 硬件 CSV 转成 tracking npz。

  Args:
    input_dir: 存放 ``pikachu_walk_*.csv`` / ``pikachu_stand.csv`` 的目录。
    output_dir: npz 输出目录(每条 clip 一个文件,文件名 = csv 的 stem)。
    input_fps: CSV 的采样率。rmg ``gait_generator.py`` 的 FPS=50。
    output_fps: 目标采样率,须与 tracking 任务的 ``1/(timestep*decimation)``
      一致(默认 0.005*4 -> 50Hz)。
    device: FK 用的设备。
  """
  in_dir = Path(input_dir)
  out_dir = Path(output_dir)
  csv_files = sorted(in_dir.glob("*.csv"))
  if not csv_files:
    raise FileNotFoundError(f"no csv under {in_dir}")

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA unavailable, falling back to CPU (slow).")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(
    SceneCfg(
      num_envs=1,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_pikachu_robot_cfg()},
    ),
    device=device,
  )
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  leg_joint_ids = list(robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)[0])
  if len(leg_joint_ids) != 10:
    raise RuntimeError(f"expected 10 leg joints, found {len(leg_joint_ids)}")

  print(f"robot joints ({len(robot.joint_names)}): {robot.joint_names}")
  print(f"leg joint ids: {leg_joint_ids}")
  _sign = left_leg_sign(robot, leg_joint_ids)
  print("left-leg sign (rmg -> MJCF):")
  for _n, _s in zip(LEG_JOINT_NAMES, _sign.tolist(), strict=True):
    print(f"  {_n:<26} {_s:+.0f}")
  print(f"bodies ({len(robot.body_names)}): {robot.body_names}")

  report: list[dict[str, Any]] = []
  for csv_path in tqdm(csv_files, desc="converting", ncols=100):
    out_path = out_dir / f"{csv_path.stem}.npz"
    info = convert_one(
      csv_path,
      out_path,
      scene=scene,
      sim=sim,
      robot=robot,
      leg_joint_ids=leg_joint_ids,
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
    )
    report.append(info)

  print("\n=== conversion summary ===")
  print(f"{'clip':42s} {'frames':>7s} {'lin_err':>9s} {'ang_err':>9s} {'nan':>5s}")
  worst_lin = 0.0
  worst_ang = 0.0
  for r in report:
    worst_lin = max(worst_lin, r["fk_lin_vel_max_err"])
    worst_ang = max(worst_ang, r["fk_ang_vel_max_err"])
    flag = "  !!  " if r["nan"] else ""
    print(
      f"{r['clip']:42s} {r['frames']:7d} {r['fk_lin_vel_max_err']:9.5f} "
      f"{r['fk_ang_vel_max_err']:9.5f} {str(r['nan']):>5s}{flag}"
    )
  print(
    f"\nclips={len(report)}  worst fk lin err={worst_lin:.5f}  ang err={worst_ang:.5f}"
  )
  if any(r["nan"] for r in report):
    raise RuntimeError("NaN detected in converted npz — aborting")

  import json

  (out_dir / "conversion_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
  )
  print(f"wrote {out_dir}/conversion_report.json")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
