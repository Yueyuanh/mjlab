"""关键帧动作可视化预览（交互式调试）。

在 MuJoCo 原生 viewer 中播放由关键帧插值出的动作轨迹，方便**调试关键帧
数值**（姿势、过渡快慢）而无需先录 CSV。支持两种播放模式：

  - ``kinematic``（默认）：每帧把插值**目标姿态**运动学写入仿真，显示
    的是"设计出来的动作"，播放快、适合微调关键帧。
  - ``pd``：与录制一致的 PD 打点执行（warp 动力学），显示"机器人实际
    能跟出来的动作"，更真实、播放较慢。

交互：**空格** 暂停/继续，关闭窗口退出，``--loop`` 循环播放。底层用
``mujoco.viewer.launch_passive`` 渲染 warp 状态同步到 ``mj_data`` 后的
画面（见 ``MocapEnv.sync_to_mj_data``）。
"""

from __future__ import annotations

import time
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.mocap.keyframes import build_trajectory
from mjlab.mocap.mocap_env import MocapEnv
from mjlab.mocap.recorder import load_keyframes_json

# GLFW 空格键码（mujoco.viewer 的 key_callback 传入 GLFW 键码）。
_KEY_SPACE = 32


def _steps_per_frame(fps: float, physics_dt: float) -> int:
  """每个预览帧推进的物理步数（decimation），同 recorder 保持一致。"""
  return max(1, round(1.0 / (fps * physics_dt)))


def _new_user_geom(scn: mujoco.MjvScene, geom_type: mujoco.mjtGeom) -> mujoco.MjvGeom:
  """分配一个填齐渲染必需字段的用户叠加几何体槽位。

  ``MjvScene`` 里未使用的 ``geoms`` 槽位字段是垃圾值；渲染器按
  ``dataid``（>=0 时视为 model 数据索引）、``objtype``、``category``
  等字段解析几何，垃圾值会把几何体直接丢弃（甚至越界崩溃）。这里统一
  初始化：不引用任何 model 数据（``dataid=-1``、``objtype=UNKNOWN``、
  ``texid=-1``、``matid=-1``），并声明 ``category=ALL`` 保证任何类别
  掩码下都可见。
  """
  g = scn.geoms[scn.ngeom]
  scn.ngeom += 1
  g.type = geom_type
  g.dataid = -1
  g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
  g.objid = -1
  g.segid = -1
  g.texid = -1
  g.matid = -1
  g.category = mujoco.mjtCatBit.mjCAT_ALL
  g.transparent = False
  g.texcoord = False
  g.texuniform = False
  g.texrepeat[:] = [1.0, 1.0]
  g.emission = 0.0
  g.shininess = 0.0
  g.reflectance = 0.0
  g.specular = 0.0
  g.camdist = 0.0
  g.modelrbound = 0.0
  g.label = b""
  return g


def _com_projection_overlay(
  viewer: mujoco.viewer.Handle,
  model: mujoco.MjModel,
  data: mujoco.MjData,
) -> None:
  """在 viewer 的叠加场景（``user_scn``）里实时画整机 COM 及地面投影。

  叠加三样东西（平面地形地面 z=0）：
    - 黄色竖线：COM → 地面投影点的竖直连线。
    - 绿点：整机质心 COM 位置（``data.subtree_com[0]``）。
    - 红点：COM 在地面的投影——判断机器人是否稳定（投影是否落在
      支撑多边形内）的参考点。

  必须在 ``sync_to_mj_data()``（即 ``mj_forward``）之后调用，此时
  ``data`` 的质心派生量已就绪。
  """
  # ``mj_forward`` 已更新 subtree_com；第 0 行（worldbody 子树）即整机质心。
  com = np.asarray(data.subtree_com[0], dtype=np.float32)
  proj = np.array([com[0], com[1], 0.0], dtype=np.float32)

  scn = viewer.user_scn
  scn.ngeom = 0  # 每帧重置叠加场景，再往里填几何体

  # 竖线：中心在中点，总长 2*size[0] = com[2]，沿世界 -z 方向。
  # mat 行主序，其第一列 (0,0,-1) 是局部 x 轴方向（= 线段方向）。
  line = _new_user_geom(scn, mujoco.mjtGeom.mjGEOM_LINE)
  line.pos[:] = 0.5 * (com + proj)
  line.size[:] = [0.5 * com[2], 0.0, 0.0]
  line.mat[:] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
  line.rgba[:] = [0.95, 0.85, 0.2, 1.0]

  # COM 位置（绿点）。
  com_pt = _new_user_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE)
  com_pt.pos[:] = com
  com_pt.size[:] = [0.012, 0.0, 0.0]
  com_pt.rgba[:] = [0.2, 0.9, 0.3, 1.0]

  # 地面投影点（红点，略微抬高避免与地面纹理交错）。
  proj_pt = _new_user_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE)
  proj_pt.pos[:] = proj
  proj_pt.pos[2] = 0.01
  proj_pt.size[:] = [0.012, 0.0, 0.0]
  proj_pt.rgba[:] = [0.95, 0.2, 0.2, 1.0]


def preview_keyframes(
  keyframes: str,
  fps: float = 30.0,
  easing: Literal["linear", "ease_in_out"] = "linear",
  mode: Literal["kinematic", "pd"] = "kinematic",
  loop: bool = False,
  fast: bool = False,
  device: str = "cpu",
  physics_dt: float = 0.005,
  write_root: bool = False,
  show_com: bool = True,
) -> None:
  """在交互 viewer 中预览一组关键帧动作。

  Args:
    keyframes: 关键帧 JSON 文件路径（格式同 ``record_cli``）。
    fps: 预览帧率（Hz）。CLI 显式传入时覆盖 JSON 里的 ``fps``。
    easing: 插值缓动，``"linear"`` 或 ``"ease_in_out"``。CLI 显式传入时
      覆盖 JSON。
    mode: 播放模式，``"kinematic"``（目标姿态）或 ``"pd"``（PD 执行）。
    loop: 播完是否循环。False 时播完停在末帧（关闭窗口退出）。
    fast: 满速播放（不做实时节流）。pd 模式每帧本身较慢，建议配合使用。
    device: 仿真设备，``"cpu"`` 或 ``"cuda:0"``。
    physics_dt: 物理仿真步长（秒），仅 pd 模式有意义。
    write_root: pd 模式是否把 root 运动学写入（钉在插值轨迹上）。默认
      False，与录制默认一致：root 由浮动基座动力学自由演化。
    show_com: 是否叠加显示整机质心 COM 及其地面投影（黄竖线 + 绿点 +
      红点）。默认 True，用于观察机器人是否稳定（COM 投影是否落在支撑
      多边形内）。

  Raises:
    RuntimeError: 当前环境无法打开原生 MuJoCo viewer（需要本地图形界面）。
  """
  env = MocapEnv(device=device, physics_dt=physics_dt)
  kf_list, cfg = load_keyframes_json(keyframes)
  if fps != 30.0:
    cfg.fps = fps
  if easing != "linear":
    cfg.easing = easing
  traj = build_trajectory(kf_list, env.joint_names, cfg=cfg)

  # 起始姿态：运动学写入第一帧，viewer 打开即可见。
  env.reset(
    joint_pos=traj.dof_pos[0],
    root_pos=traj.root_pos[0],
    root_quat_xyzw=traj.root_quat_xyzw[0],
  )
  env.sync_to_mj_data()

  steps_per_frame = _steps_per_frame(cfg.fps, physics_dt)
  frame_interval = 1.0 / cfg.fps if cfg.fps > 0 else 0.0

  if mode == "pd":

    def apply(frame: int) -> None:
      env.pd_step(
        joint_target=traj.dof_pos[frame],
        root_pos=traj.root_pos[frame],
        root_quat_xyzw=traj.root_quat_xyzw[frame],
        n_steps=steps_per_frame,
        write_root=write_root,
      )
      env.sync_to_mj_data()
  else:

    def apply(frame: int) -> None:
      env.reset(
        joint_pos=traj.dof_pos[frame],
        root_pos=traj.root_pos[frame],
        root_quat_xyzw=traj.root_quat_xyzw[frame],
      )
      env.sync_to_mj_data()

  paused: list[bool] = [False]

  def on_key(keycode: int) -> None:
    if keycode == _KEY_SPACE:
      paused[0] = not paused[0]
      print("[preview] 已暂停（空格继续）" if paused[0] else "[preview] 继续播放")

  try:
    viewer = mujoco.viewer.launch_passive(
      env.mj_model,
      env.mj_data,
      key_callback=on_key,
      show_left_ui=True,
      show_right_ui=True,
    )
  except Exception as e:  # noqa: BLE001 - 需要把底层失败转成可读提示
    raise RuntimeError(
      "无法打开原生 MuJoCo viewer（需要本地图形界面）。"
      "若在无显示环境，可用 --mode kinematic 配合 --render 方案代替。"
    ) from e

  print(
    f"[preview] 播放 {traj.num_frames} 帧 (mode={mode}, "
    f"{cfg.fps:.1f}fps, easing={cfg.easing})。"
    f"空格=暂停/继续，关闭窗口=退出{'，播完自动循环' if loop else ''}。"
  )

  frame = 0
  try:
    while viewer.is_running():
      if paused[0]:
        if show_com:
          _com_projection_overlay(viewer, env.mj_model, env.mj_data)
        viewer.sync()
        time.sleep(0.03)
        continue
      if frame >= traj.num_frames:
        if not loop:
          break
        frame = 0
      apply(frame)
      if show_com:
        _com_projection_overlay(viewer, env.mj_model, env.mj_data)
      viewer.sync()
      frame += 1
      if not fast and frame_interval > 0:
        time.sleep(frame_interval)
  finally:
    viewer.close()
