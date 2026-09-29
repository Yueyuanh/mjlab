"""判定参考 CSV 的左腿符号:候选符号表逐个跑 FK,看哪套能让两条腿物理自洽。

判据:正确的符号下,左右**踝**相对 base 的高度范围应当量级相同(都是一条腿,
只是相位相反);符号错的那些关节会把腿折进身体里,该侧踝高度范围会明显不同。

用法(在 mjlab 目录下)::

    MUJOCO_GL=egl uv run python src/mjlab/scripts/probe_leg_sign.py
"""

from __future__ import annotations

import importlib.util

import torch

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import LEG_JOINT_NAMES
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_lite import (
  get_pikachu_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

_spec = importlib.util.spec_from_file_location(
  "conv", MJLAB_SRC_PATH / "scripts" / "csv_to_npz_rmgamp.py"
)
assert _spec and _spec.loader
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

CSV = (
  MJLAB_SRC_PATH
  / "mocap"
  / "csv"
  / "RMG_AMP"
  / "csv"
  / "pikachu_walk_0.185_0.0_0.0.csv"
)

# 左腿 5 关节在 LEG_JOINT_NAMES 里的下标(0..4 = 左腿)。
L = list(LEG_JOINT_NAMES[:5])
CANDIDATES: dict[str, set[str]] = {
  "C0 不翻(现状)": set(),
  "C1 翻 pitch/knee/ankle": {L[0], L[3], L[4]},
  "C2 翻整条左腿": set(L),
  "C3 只翻 knee": {L[3]},
  "C4 翻 knee+ankle": {L[3], L[4]},
  "C5 翻 pitch+knee": {L[0], L[3]},
}


def main() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / 50.0
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
  leg_ids = list(robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)[0])
  bodies = list(robot.body_names)
  li = bodies.index("left_ankle_link")
  ri = bodies.index("right_ankle_link")
  bi = bodies.index("base_link") if "base_link" in bodies else 0

  print(f"base={bodies[bi]}  left_ankle={li}  right_ankle={ri}\n")
  print(f"{'候选':<24} {'左踝z范围(相对base)':<26} {'右踝z范围':<26} {'对称差'}")
  for name, flip in CANDIDATES.items():
    motion = conv.RmgAmpMotionLoader(
      motion_file=str(CSV), input_fps=50.0, output_fps=50.0, device=device
    )
    sign = torch.ones(len(LEG_JOINT_NAMES), device=device)
    for j in flip:
      sign[LEG_JOINT_NAMES.index(j)] = -1.0

    lz, rz, refs = [], [], []
    scene.reset()
    for _ in range(400):
      (state, reset_flag) = motion.get_next_state()
      (
        base_pos,
        base_rot,
        base_lv,
        base_av,
        dof_pos,
        dof_vel,
      ) = state
      rs = robot.data.default_root_state.clone()
      rs[:, 0:3] = base_pos
      rs[:, 3:7] = base_rot
      rs[:, 7:10] = base_lv
      rs[:, 10:] = base_av
      robot.write_root_state_to_sim(rs)
      jp = robot.data.default_joint_pos.clone()
      jv = robot.data.default_joint_vel.clone()
      refs.append(dof_pos[0].detach().cpu())
      jp[:, leg_ids] = dof_pos * sign
      jv[:, leg_ids] = dof_vel * sign
      robot.write_joint_state_to_sim(jp, jv)
      sim.forward()
      scene.update(sim.mj_model.opt.timestep)
      bp = robot.data.body_link_pos_w[0]
      lz.append((bp[bi, 2] - bp[li, 2]).item())
      rz.append((bp[bi, 2] - bp[ri, 2]).item())
      if reset_flag:
        break

    lm, lM = min(lz), max(lz)
    rm, rM = min(rz), max(rz)
    sym = abs((lm + lM) / 2 - (rm + rM) / 2)
    # 限位符合度:参考落在 MJCF 关节限位外的比例(正确符号下应为 0)。
    lim = robot.data.joint_pos_limits[0][leg_ids]
    ref = torch.stack(refs).to(device) * sign  # (T, 10)
    bad = ((ref < lim[:, 0] - 1e-6) | (ref > lim[:, 1] + 1e-6)).float()
    n_bad = int(bad.any(dim=1).sum())
    worst = torch.clamp(lim[:, 0] - ref, min=0) + torch.clamp(ref - lim[:, 1], min=0)
    print(
      f"{name:<24} [{lm:+.3f}, {lM:+.3f}] span={lM - lm:.3f}   "
      f"[{rm:+.3f}, {rM:+.3f}] span={rM - rm:.3f}   {sym:.4f}   "
      f"越界帧 {n_bad:>3}/{len(lz)}  左腿最大越界 "
      f"{worst[:, :5].max().item():.3f}  右腿最大越界 {worst[:, 5:].max().item():.3f}"
    )


if __name__ == "__main__":
  main()
