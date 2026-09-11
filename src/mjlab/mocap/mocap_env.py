"""Pikachu MuJoCo 仿真环境（数据录制用）。

基于 mjlab 的 ``Scene`` + ``Simulation``（Warp 后端，CPU/GPU 均可），
加载 Pikachu 机器人（``get_pikachu_robot_cfg``，FULL_COLLISION_WITHOUT_SELF
碰撞配置）与平面地形，提供 PD 打点录制所需的仿真接口：

  - ``reset``：把 root 位姿 + 关节角**运动学**写入仿真（无 PD 过渡），
    用于在动作开始前摆放初始姿态。
  - ``pd_step``：写入关节 PD 目标（ctrl，单位弧度），然后推进若干物理
    步。默认不写 root，让浮动基座由动力学与接触自然演化；需要兼容旧的
    root-guided 调试路径时可显式打开 root 运动学写入。
  - ``get_state``：读取仿真实际状态，返回
    ``(root_pos, root_quat_xyzw, dof_pos)``——与 ``csv_to_npz.py``
    的输入列格式完全对齐（21 列 = root_pos 3 + root_quat 4 + dof 14）。

关节/执行器命名约定（务必注意）：
  - ``robot.joint_names``（带 ``_joint`` 后缀）= ``robot.data.joint_pos``
    的列序 = CSV dof 列序（与 TRACKING_FLOW §3.3 一致）。
  - ``robot.actuator_names`` 是 **ctrl 槽位顺序**，与 joint_names 不同
    （J4310 的髋 pitch / 膝排在最前）。写 ctrl 时通过
    ``_joint_to_actuator_idx`` 重排。
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots import get_pikachu_robot_cfg
from mjlab.entity import Entity, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg


def scale_pd_gains(robot_cfg: EntityCfg, multiplier: float) -> EntityCfg:
  """返回把各执行器 Kp/Kd 统一乘以 ``multiplier`` 的配置副本。

  仅用于 mocap 预览/录制时调硬 PD 跟随（预算外部不修改原配置，也不碰
  训练共享的执行器定义）。对含 ``stiffness``/``damping`` 的执行器 cfg 用
  ``dataclasses.replace`` 生成新实例，其余执行器原样保留。``multiplier==1.0``
  时原样返回入参。

  Args:
    robot_cfg: 机器人实体配置（含 articulation 执行器）。
    multiplier: PD 增益倍率（>0）。1.0 表示不缩放。

  Raises:
    ValueError: 配置缺 articulation，无从缩放。
  """
  if multiplier == 1.0:
    return robot_cfg
  art = robot_cfg.articulation
  if art is None:
    raise ValueError("robot_cfg 缺少 articulation，无法缩放 PD 增益")

  def _scaled(act: object) -> object:
    if hasattr(act, "stiffness") and hasattr(act, "damping"):
      return dataclasses.replace(
        act,
        stiffness=act.stiffness * multiplier,
        damping=act.damping * multiplier,
      )
    return act

  new_art = dataclasses.replace(art, actuators=tuple(_scaled(a) for a in art.actuators))
  return dataclasses.replace(robot_cfg, articulation=new_art)


class MocapEnv:
  """单环境 Pikachu 仿真，用于 PD 打点 + 插值录制。

  Args:
    device: 仿真设备（``"cpu"`` 或 ``"cuda:0"``）。默认 ``"cpu"``，
      无需 GPU 也能录制。
    physics_dt: 物理仿真步长（秒）。默认 0.005，与 tracking env 一致。
    robot_cfg: 机器人实体配置，默认 14dof（``get_pikachu_robot_cfg()``）；
      传 ``get_pikachu_27dof_mocap_robot_cfg()`` 可预览/录制 27dof 机器人。
    pd_gain_multiplier: 把执行器 Kp/Kd 统一放大的倍率（映射缩小预览/录制
      用的 PD 硬度）。默认 1.0 不缩放；觉得跟随太软可调大到 1.5~3。
      只影响本 mocap 环境，训练共享常数不受影响。
  """

  def __init__(
    self,
    device: str = "cpu",
    physics_dt: float = 0.005,
    robot_cfg: EntityCfg | None = None,
    pd_gain_multiplier: float = 1.0,
  ):
    self.device = device
    self.physics_dt = physics_dt

    if robot_cfg is None:
      robot_cfg = get_pikachu_robot_cfg()
    if pd_gain_multiplier != 1.0:
      robot_cfg = scale_pd_gains(robot_cfg, pd_gain_multiplier)
    scene_cfg = SceneCfg(
      num_envs=1,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    )
    self._scene = Scene(scene_cfg, device=device)
    model = self._scene.compile()
    sim_cfg = SimulationCfg(mujoco=MujocoCfg(timestep=physics_dt))
    self._sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    self._scene.initialize(self._sim.mj_model, self._sim.model, self._sim.data)
    self._scene.reset()

    self.robot: Entity = self._scene["robot"]
    self.joint_names: tuple[str, ...] = self.robot.joint_names
    self.actuator_names: tuple[str, ...] = self.robot.actuator_names

    # 软限位（0.9 × 硬限位），形状 (D, 2)，D = 关节数，joint_names 序。
    self.soft_joint_pos_limits: np.ndarray = (
      self.robot.data.soft_joint_pos_limits[0].cpu().numpy()
    )

    # actuator_names 序 → joint_names 序的索引；用于把 joint target 重排到 ctrl。
    self._actuator_to_joint_idx = np.array(
      [self.joint_names.index(name) for name in self.actuator_names],
      dtype=np.int64,
    )

    # 浮动基默认初始状态（init_state，KNEES_BENT：正立、z=0.14、速度 0）。
    # ``scene.reset()`` 只清状态，不写 root；而 ``mj_model.qpos0`` 的 free
    # joint 段是编译默认（可能陷地/翻转），不可信。这里显式用
    # ``default_root_state`` 初始化，否则"仅配关节角"的录制会从坏初始姿态
    # 起跑（机器人弹飞/倾倒）。
    self.robot.write_root_state_to_sim(self.robot.data.default_root_state)
    self._sim.forward()

  # -- 预览渲染 ----------------------------------------------------------

  @property
  def mj_model(self) -> mujoco.MjModel:
    """原生 MuJoCo 模型（预览/回放渲染用）。"""
    return self._sim.mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    """原生 MuJoCo 数据（预览/回放渲染用）。

    注意：仿真推进发生在 warp 侧，``mj_data`` 不会自动更新，需要先调用
    ``sync_to_mj_data`` 把 warp 最新状态同步进来再渲染。
    """
    return self._sim.mj_data

  def sync_to_mj_data(self) -> None:
    """把 warp 当前状态同步到 ``mj_data`` 并刷新派生量，供渲染。

    预览循环里先 PD 打点（``pd_step``）或运动学写入（``reset``），再调
    用本方法，``mujoco.viewer`` 就能显示最新一帧。
    """
    data = self._sim.data
    mj_data = self.mj_data
    mj_data.qpos[:] = data.qpos[0].cpu().numpy()
    mj_data.qvel[:] = data.qvel[0].cpu().numpy()
    mujoco.mj_forward(self.mj_model, mj_data)

  # -- 底层写入 ----------------------------------------------------------

  def _pose7(self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray) -> torch.Tensor:
    """组 root pose 张量 ``(1, 7)``：pos(3) + quat **wxyz**(4)。"""
    quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]
    pose = np.concatenate([np.asarray(root_pos, float), quat_wxyz])
    return torch.as_tensor(pose[None], dtype=torch.float32, device=self.device)

  def _target_to_ctrl(
    self, joint_target: dict[str, float] | np.ndarray
  ) -> torch.Tensor:
    """把关节目标（joint_names 序）重排为 ctrl（actuator_names 序）。

    Args:
      joint_target: 关节名→角度（弧度）的 dict，或 ``(D,)`` 数组
        （joint_names 序，D = 关节数）。

    Returns:
      ``(1, D)`` 张量，槽位顺序 = actuator_names（= ctrl 槽位）。
    """
    if isinstance(joint_target, dict):
      # dict 必须提供全部关节（缺省会 KeyError，比静默填 0 更安全）。
      vals = np.array([joint_target[name] for name in self.joint_names], dtype=float)
    else:
      vals = np.asarray(joint_target, dtype=float)
      if vals.shape != (len(self.joint_names),):
        raise ValueError(
          f"joint_target 形状应为 ({len(self.joint_names)},)，收到 {vals.shape}"
        )
    ctrl = vals[self._actuator_to_joint_idx]
    return torch.as_tensor(ctrl[None], dtype=torch.float32, device=self.device)

  # -- 对外接口 ----------------------------------------------------------

  def reset(
    self,
    joint_pos: np.ndarray | dict[str, float] | None = None,
    root_pos: np.ndarray | None = None,
    root_quat_xyzw: np.ndarray | None = None,
  ) -> None:
    """运动学写入初始状态（不经过 PD 过渡），并刷新派生量。

    用于在录制开始前把机器人摆到第一个关键帧的姿态。

    Args:
      joint_pos: ``(D,)`` 数组（joint_names 序）或关节名→角度 dict。
        None 表示保持当前关节角。
      root_pos: ``(3,)`` 世界系位置。None 表示保持当前位置。
      root_quat_xyzw: ``(4,)`` 世界系四元数（**xyzw 序**）。
        None 表示保持当前姿态。
    """
    if joint_pos is not None:
      if isinstance(joint_pos, dict):
        # dict 必须提供全部关节（缺省会 KeyError）。
        joint_pos = np.array(
          [joint_pos[name] for name in self.joint_names], dtype=float
        )
      self.robot.write_joint_position_to_sim(
        torch.as_tensor(joint_pos[None], dtype=torch.float32, device=self.device)
      )
    if root_pos is not None and root_quat_xyzw is not None:
      self.robot.write_root_link_pose_to_sim(self._pose7(root_pos, root_quat_xyzw))
    self._sim.forward()

  def pd_step(
    self,
    joint_target: dict[str, float] | np.ndarray,
    root_pos: np.ndarray | None = None,
    root_quat_xyzw: np.ndarray | None = None,
    n_steps: int = 1,
    write_root: bool = False,
  ) -> None:
    """一个录制帧：写入 PD 目标 + 可选 root 姿态，推进 ``n_steps`` 物理步。

    PD 打点的核心：把插值得到的关节目标角写入 ctrl（MuJoCo position
    actuator 内部做 PD），让机器人真实动力学响应若干步后读取状态。

    Args:
      joint_target: 关节 PD 目标（joint_names 序），见 ``_target_to_ctrl``。
      root_pos / root_quat_xyzw: 本帧 root 运动学姿态（xyzw 序）。
        仅当 ``write_root=True`` 且两者同时给出时每子步前写回，用于兼容
        旧的 root-guided 调试/录制路径。
      n_steps: 本帧推进的物理步数（= decimation）。
      write_root: 是否写回 root。默认 False，让浮动基座自由演化。
    """
    ctrl = self._target_to_ctrl(joint_target)
    for _ in range(n_steps):
      self.robot.write_ctrl_to_sim(ctrl)
      if write_root and root_pos is not None and root_quat_xyzw is not None:
        self.robot.write_root_link_pose_to_sim(self._pose7(root_pos, root_quat_xyzw))
      self._sim.step()

  def get_state(
    self,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取仿真实际状态。

    Returns:
      ``(root_pos, root_quat_xyzw, dof_pos)``：
        - ``root_pos`` ``(3,)`` 世界系位置（米）
        - ``root_quat_xyzw`` ``(4,)`` 世界系四元数，**xyzw 序**
        - ``dof_pos`` ``(D,)`` 关节角（弧度），**joint_names 序**
      前 7 个值与后 14 个值拼接即为 CSV 的每一行（21 列）。
    """
    pose = self.robot.data.root_link_pose_w[0].cpu().numpy()
    root_pos = pose[:3].copy()
    root_quat_wxyz = pose[3:7]
    root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]
    dof_pos = self.robot.data.joint_pos[0].cpu().numpy()
    return root_pos, root_quat_xyzw, dof_pos

  # -- 工具 --------------------------------------------------------------

  def check_joint_limits(
    self, dof_pos: np.ndarray, joint_names: str = ""
  ) -> tuple[bool, np.ndarray]:
    """检查一组关节角是否落在软限位内。

    Args:
      dof_pos: ``(D,)`` 或 ``(N, D)`` 关节角（joint_names 序，D = 关节数）。
      joint_names: 仅用于报错信息里的标识（例如"关键帧"）。

    Returns:
      ``(ok, margins)``：``ok`` 是否全部在限位内；``margins`` 为每个关节
      距最近限位的距离（``(D,)`` 或 ``(N, D)``，负值表示越界）。
    """
    arr = np.asarray(dof_pos, dtype=float)
    lo = self.soft_joint_pos_limits[:, 0]
    hi = self.soft_joint_pos_limits[:, 1]
    margins = np.minimum(arr - lo, hi - arr)
    ok = bool(np.all(margins >= 0))
    if not ok:
      for i, name in enumerate(self.joint_names):
        bad = np.any(arr[..., i] < lo[i]) or np.any(arr[..., i] > hi[i])
        if bad:
          print(
            f"[WARN] {joint_names}关节 '{name}' 超出软限位 [{lo[i]:.3f}, {hi[i]:.3f}]"
          )
    return ok, margins
