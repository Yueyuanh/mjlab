"""把 14dof 四足 mocap CSV 重放成 **27dof 四足 tracking 参考** npz。

复用 ``csv_to_npz_pikachu`` 的 ``MotionLoader``(CSV 加载/插值/差分),但把场景换成
**27dof 四足**机器人(``get_pikachu_27dof_quad_robot_cfg``):root+身体状态写入 27dof
四足的 root,14 个录制关节(腿 10 + 臂 pitch/roll,见 ``PIKACHU_JOINT_NAMES``)写入
对应关节,其余含 elbow/arm_yaw/头/耳/尾 保持四足关键帧默认。FK 前向后 record 全量
body 状态 → 产出 tracking ``MotionLoader`` 需要的 ``body_pos_w / body_quat_w /
body_lin_vel_w / body_ang_vel_w``(现存 ``pikachu_quad.npz`` 缺这些键,正是 tracking
play 崩溃的原因)。

**elbow 告警**:录制只驱动臂 pitch/roll,**elbow(前腿膝盖)保持关键帧默认**——若
bounding 需要前腿屈膝,需在 mocap 里另录 elbow,或录制前在关键帧配平肘角。

用法(输入须为 21 列 = root_pos 3 + root_quat **xyzw** 4 + dof_pos 14 的 CSV):
```sh
uv run python -m mjlab.scripts.make_27dof_quad_reference \
    --input-file src/mjlab/mocap/npz/pikachu_walk_27dof_npz/pikachu_quad.csv \
    --output-name pikachu_quad_27dof \
    --input-fps 30 --output-fps 50 --device cpu
```
``--output-name`` 决定 npz 落在 ``mocap/npz/{output_name}.npz``。

把产出路径回填到 ``pikachu_27dof_quad_constants.QUAD_27DOF_MOTION_FILE`` 替换占位。
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.pikachu_xml.pikachu_27dof_quad_constants import (
  QUAD_27DOF_WALK_JOINT_NAMES,
  get_pikachu_27dof_quad_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.scripts.csv_to_npz_pikachu import PIKACHU_JOINT_NAMES, MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

REC_JOINT_NAMES: tuple[str, ...] = (
  PIKACHU_JOINT_NAMES  # 录制 14 dof(腿+臂 pitch/roll)。
)


def main(
  input_file: str,
  output_name: str = "motion",
  input_fps: int = 30,
  output_fps: int = 50,
  device: str = "cpu",
  output_file: str | None = None,
  z_bias: float = 0.0,
) -> None:
  """Replay a 14dof quad CSV onto the 27dof quad robot and save a full tracking npz.

  Args:
    z_bias: 对整段参考 ``body_pos_w`` 第三列统一加的 z 偏置(米)。录制常把脚沉到
      地面以下(本 clip 最低点 ≈ -0.146),给正值可整体抬离地面;相对肢体几何不变。
  """
  if output_file is None:
    out_dir = Path(__file__).resolve().parent.parent / "mocap" / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(out_dir / f"{output_name}.npz")

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING] CUDA not available, falling back to CPU.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(
    SceneCfg(
      num_envs=1,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_pikachu_27dof_quad_robot_cfg()},
    ),
    device=device,
  )
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )
  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(REC_JOINT_NAMES, preserve_order=True)[0]
  # 参考 joint_pos/vel 列序必须 == cfg.joint_names(= QUAD_27DOF_WALK_JOINT_NAMES)。
  walk_indexes = robot.find_joints(QUAD_27DOF_WALK_JOINT_NAMES, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [np.array(output_fps)],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  steps = motion.output_frames
  h0 = float(robot.data.default_root_state[0, 2].detach())  # 四足起步站高
  for _ in range(steps):
    (
      (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel),
      _reset,
    ) = motion.get_next_state()

    root = robot.data.default_root_state.clone()
    root[:, 0:3] = base_pos
    root[:, 2] = (
      h0  # z 用四足默认站高，忽略 csv 绝对 z → 避免低位穿地/narrowphase overflow
    )
    root[:, :2] += scene.env_origins[:, :2]
    root[:, 3:7] = base_rot
    root[:, 7:10] = base_lin_vel
    root[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = dof_pos
    joint_vel[:, robot_joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, walk_indexes].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, walk_indexes].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )

  for k in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[k] = np.stack(log[k], axis=0)

  if z_bias != 0.0:
    log["body_pos_w"] = log["body_pos_w"].copy()
    log["body_pos_w"][:, :, 2] += z_bias
    print(f"applied z_bias = {z_bias:+g} m to body_pos_w[:, :, 2]")

  np.savez_compressed(output_file, **log)
  n_body = log["body_pos_w"].shape[1]
  print(f"wrote {output_file}")
  print(
    f"joint_pos/vel = {log['joint_pos'].shape}, body_pos_w = "
    f"{log['body_pos_w'].shape} (T,{n_body},3), fps = {output_fps}"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
