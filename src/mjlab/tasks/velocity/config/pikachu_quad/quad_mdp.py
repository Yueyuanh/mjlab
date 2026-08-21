"""Pikachu 四足私有 mdp:四足前方系命令/速度观测、带偏角朝向奖励、摔倒终止。

独立于共享 ``velocity/mdp``(不动共享 ``upright``/``bad_orientation``/
``track_linear_velocity`` 等)。

四足采用 **base 前俯**(``QUAD_START`` rot 前俯 ≈90°)的爬行 morphology,共享
base-frame 速度语义失效:base+X 被竖起来,共享 reward/obs 的 ``vel_command_b``
比 ``root_link_lin_vel_b`` 不再沿世界水平。四足统一到 **四足前方系**:
  命令(``vel_command_w``)与反馈(``root_link_lin_vel_w``)都绕世界竖直旋转 Φ,
  使背轴(+Z,``QUAD_BODY_AXIS``)水平投影指向 +X → 策略看到"相对四足前方的
  前进/横移/转向"。Φ 正交,差范数与在世界系一致,只是换 frame 表达。
- ``upright`` 退回带 ``tilt_deg`` 偏角:奖励峰在躯干背轴与世界 up 夹角 ≈
  ``tilt_deg``(= 四足前俯角),而不是竖直(=0)。``fell_over`` 同理:背轴偏离
  ``tilt_deg`` 超过限位即终止。
- viser 摇杆(shared ``compute`` 只写 ``vel_command_b``)在四足子类里转写到
  ``vel_command_w``,与观测/奖励同帧,推摇杆才能被策略看到。

保留 debug_vis:从 base 出发画两根箭头 —— **橙=当前背轴**、**紫=目标
(按 ``tilt_deg`` 转好)方向**,夹角即当前对四足目标姿态的偏离。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
  "QUAD_TILT_DEG",
  "QUAD_BODY_AXIS",
  "quad_track_linear_velocity",
  "quad_track_angular_velocity",
  "quad_front_command",
  "quad_front_lin_vel",
  "quad_front_ang_vel",
  "quad_reset_root_state_uniform",
  "quad_upright",
  "quad_bad_orientation",
  "QuadUniformVelocityCommand",
  "QuadUniformVelocityCommandCfg",
]

# 四足 reset 的 yaw 随机范围(弧度,绕**世界竖直**轴旋转朝向)。
_QUAD_RESET_YAW = (-math.pi, math.pi)

# 四足 target 躯干前俯角(度):``QUAD_BODY_AXIS``(背轴 +Z)与世界 up 的目标夹角。
# 门禁标定(训练 cfg /tmp/quad_diag.py):zero-action 120 步 settle 全程背轴 up 夹角
# = 110.1°(≈水平再俯 20°),reset→settle 恒定。viewer 橙/紫箭头应重合于羽;若调
# 前俯角(XML/QUAD_KEYFRAME)需重标定此值。
QUAD_TILT_DEG: float = 110.0

# 四足"背轴"在当地机体的方向:机体 +Z。
QUAD_BODY_AXIS: tuple[float, float, float] = (0.0, 0.0, 1.0)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def quad_track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """四足前方系速度跟踪:reward 峰在实际速度(四足前方系)≈ 命令(四足前方系)。

  Φ 正交 → 差范数与在世界系一致;统一到前方系是让观测/奖励同 frame,并避开共享
  ``vel_command_b`` vs ``root_link_lin_vel_b`` 在 base 前俯下坐标不一致的问题。
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, UniformVelocityCommand)
  vel_w = command.vel_command_w  # [B,3] 世界期望速度
  actual_w = asset.data.root_link_lin_vel_w  # [B,3] 世界实际速度
  fx, fy = _quad_front_axes(env)
  cf, cs = _to_front(vel_w[:, :2], fx, fy)
  af, ac = _to_front(actual_w[:, :2], fx, fy)
  xy_err = torch.square(cf - af) + torch.square(cs - ac)
  z_err = torch.square(actual_w[:, 2])
  return torch.exp(-(xy_err + z_err) / std**2)


def quad_track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """四足世界系偏航角速度跟踪(与共享版结构同构,方向取世界)。"""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, UniformVelocityCommand)
  vel_w = command.vel_command_w  # [B,3] 世界系 [x,y,ang_z]
  actual_w = asset.data.root_link_ang_vel_w  # [B,3] 世界系角速度
  z_err = torch.square(vel_w[:, 2] - actual_w[:, 2])
  xy_err = torch.sum(torch.square(actual_w[:, :2]), dim=1)
  return torch.exp(-(z_err + xy_err) / std**2)


def _quad_front_axes(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
  """四足前方参考系在世界的 [fx, fy]:背轴(+Z)的世界水平投影,归一化。

  绕 base 前俯后的完整姿态取投影会退回 base 局向;这里只取背轴在世界**水平面**的
  分量(爬行背朝前),归一后即"四足前方"。base 前俯时背轴世界分量近水平、数值健壮;
  若背轴竖直(前俯→直立)norm→0,clamp 兜底但仍附返回旧前进方向。
  """
  asset: Entity = env.scene["robot"]
  back_w = quat_apply(
    asset.data.root_link_quat_w,
    torch.tensor(QUAD_BODY_AXIS, device=env.device).expand_as(
      asset.data.root_link_quat_w[:, :3]
    ),
  )  # [B,3] 背轴世界方向
  hx, hy = back_w[:, 0], back_w[:, 1]
  norm = torch.sqrt(hx * hx + hy * hy).clamp_min(1e-6)
  return hx / norm, hy / norm


def _to_front(
  vec: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """把世界水平向量 [vx, vy] 转到四足前方系:[前方, 侧向] = Φ·vec。

  旋转矩阵 Φ = [[fx, fy], [-fy, fx]](绕世界竖直,使背轴指向 +X)。Φ 正交,因此
  命令/反馈转到前方系后差范数不变,但观测从此表达为"相对四足前方",与 base 前俯解耦。
  """
  fwd = fx * vec[:, 0] + fy * vec[:, 1]
  strafe = -fy * vec[:, 0] + fx * vec[:, 1]
  return fwd, strafe


def quad_reset_root_state_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """四足 reset:绕**世界竖直**随机朝向,保持前俯爬行姿态。

  共享 ``reset_root_state_uniform`` 用 ``quat_mul(default, delta)`` 把 yaw 加在
  base 局部帧;四足前俯后 base 局部 z 贴近水平,该 yaw 随机会变成绕世界水平轴翻滚
  → 初始侧躺/朝天。四足版把 roll/pitch 固定为 0、yaw 作为**世界竖直**旋转叠加在
  基础姿态(``QUAD_KEYFRAME`` 前俯)之上:任何朝向都保持"脚贴地、背水平",只转朝向。
  位置/速度随机沿用共享逻辑。
  """
  asset: Entity = env.scene[asset_cfg.name]
  device = env.device
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None

  if env_ids is None:
    env_ids = torch.arange(default_root_state.shape[0], device=device)
  n = env_ids.numel()

  # 位置:相对 default 增量 + 每 env 网格原点。
  pos = torch.zeros(n, 3, device=device)
  for i, axis in enumerate(("x", "y", "z")):
    lo, hi = pose_range.get(axis, (0.0, 0.0))
    pos[:, i] = torch.rand(n, device=device) * (hi - lo) + lo
  positions = default_root_state[env_ids, 0:3] + pos + env.scene.env_origins[env_ids]

  # 朝向:roll/pitch=0,只绕**世界竖直** z 加 yaw,叠加在基础前俯姿态之上。
  yaw_lo, yaw_hi = _QUAD_RESET_YAW
  yaw = torch.rand(n, device=device) * (yaw_hi - yaw_lo) + yaw_lo
  q_yaw_world = quat_from_euler_xyz(
    torch.zeros(n, device=device), torch.zeros(n, device=device), yaw
  )  # 绕世界 z 的外旋 quat
  orientations = quat_mul(q_yaw_world, default_root_state[env_ids, 3:7])

  # 速度:相对 default 增量。
  vel_samples = torch.zeros(n, 6, device=device)
  for i, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
    lo, hi = (velocity_range or {}).get(axis, (0.0, 0.0))
    vel_samples[:, i] = torch.rand(n, device=device) * (hi - lo) + lo
  velocities = default_root_state[env_ids, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def quad_front_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """四足前方系命令观测:[前方, 侧向, 转向角速度]。

  共享 ``generated_commands`` 返回 ``vel_command_b``(base-body,含前俯污染)。
  这里读世界命令 ``vel_command_w`` 并转到四足前方系:策略看到的是"相对四足前方的
  前进/横移/转向",不再被 base 前俯扭曲,也与下方速度反馈/奖励同 frame。
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, UniformVelocityCommand)
  vel_w = term.vel_command_w  # [B,3] wrt 世界水平 x/y + 世界角速度 z
  fx, fy = _quad_front_axes(env)
  fwd, strafe = _to_front(vel_w[:, :2], fx, fy)
  return torch.stack([fwd, strafe, vel_w[:, 2]], dim=1)


def quad_front_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """四足前方系线速度观测:[前方, 侧向, 竖直]。顶替被前俯污染的 body imu。"""
  asset: Entity = env.scene["robot"]
  vw = asset.data.root_link_lin_vel_w  # [B,3]
  fx, fy = _quad_front_axes(env)
  fwd, strafe = _to_front(vw[:, :2], fx, fy)
  return torch.stack([fwd, strafe, vw[:, 2]], dim=1)


def quad_front_ang_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """四足前方系角速度观测:绕世界竖直不变,前方系直接取世界角速度。"""
  asset: Entity = env.scene["robot"]
  return asset.data.root_link_ang_vel_w  # [B,3]


class QuadUniformVelocityCommand(UniformVelocityCommand):
  """四足命令:摇杆前方系写入 ``vel_command_w`` + 世界系 debug_vis。

  共享 ``UniformVelocityCommand`` 里摇杆只写 ``vel_command_b``、debug_vis 只画
  base-frame,两者在 base 前俯下都失真。四足覆盖:
    - ``compute``:摇杆滑块按"相对四足前方"解释,经 Φ⁻¹ 转回世界写
      ``vel_command_w``(四足观测/奖励同帧);训练路径继承父类,摇杆禁用时不变。
    - ``_debug_vis_impl``:直接画 ``vel_command_w`` 与 ``root_link_*_vel_w``
      (世界系),方向即世界意图,不经过 base 姿态。
  采样/更新/重置逻辑全部继承父类。
  """

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    """摇杆命令写入四足前方系,再经 Φ⁻¹ 转回世界 ``vel_command_w``。

    共享 ``UniformVelocityCommand.compute`` 把摇杆滑块直接写 ``vel_command_b``
    (base 局部,含前俯污染),而四足观测/奖励读 ``vel_command_w`` — 二者通道错位,
    viser 里使能 commands twist 推摇杆,策略根本看不到。这里在 super() 之后把
    滑块解释为"相对四足前方"速度并转回世界写 ``vel_command_w``,使四足观测/奖励
    与摇杆一致(``quad_front_command`` 读回前方系即得滑块值)。训练路径(摇杆禁用)
    完全继承父类,不受影响。
    """
    super().compute(dt, env_ids)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      v_f = float(self._joystick_sliders[0].value)  # 前方系前进(沿四足前方)
      v_s = float(self._joystick_sliders[1].value)  # 前方系侧向
      w = float(self._joystick_sliders[2].value)  # 转向角速度
      # 前方系 → 世界:Φ⁻¹ = Φᵀ,即 [vx, vy] = [fx -fy; fy fx]·[v_f, v_s]。
      # 取当前 env 前方轴,与 quad_front_command/quad_track_linear_velocity 同帧。
      fx, fy = _quad_front_axes(self._env)
      self.vel_command_w[idx, 0] = fx[idx] * v_f - fy[idx] * v_s
      self.vel_command_w[idx, 1] = fy[idx] * v_f + fx[idx] * v_s
      self.vel_command_w[idx, 2] = w

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.vel_command_w.cpu().numpy()
    base_pos = self.robot.data.root_link_pos_w.cpu().numpy()
    lin_vel_w = self.robot.data.root_link_lin_vel_w.cpu().numpy()
    ang_vel_w = self.robot.data.root_link_ang_vel_w.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      origin = base_pos[batch] + np.array([0.0, 0.0, z_offset * scale])
      if np.linalg.norm(base_pos[batch]) < 1e-6:
        continue
      cmd = cmds[batch]
      lv = lin_vel_w[batch]
      av = ang_vel_w[batch]

      visualizer.add_arrow(
        origin,
        origin + np.array([cmd[0], cmd[1], 0.0]) * scale,
        color=(0.2, 0.2, 0.6, 0.6),  # 蓝:命令线速(世界)
        width=0.015,
      )
      visualizer.add_arrow(
        origin,
        origin + np.array([0.0, 0.0, cmd[2]]) * scale,
        color=(0.2, 0.6, 0.2, 0.6),  # 绿:命令角速(世界)
        width=0.015,
      )
      visualizer.add_arrow(
        origin,
        origin + np.array([lv[0], lv[1], 0.0]) * scale,
        color=(0.0, 0.6, 1.0, 0.7),  # 青:实际线速(世界)
        width=0.015,
      )
      visualizer.add_arrow(
        origin,
        origin + np.array([0.0, 0.0, av[2]]) * scale,
        color=(0.0, 1.0, 0.4, 0.7),  # 浅绿:实际角速(世界)
        width=0.015,
      )


class QuadUniformVelocityCommandCfg(UniformVelocityCommandCfg):
  """四足命令配置:build 返回世界 debug_vis 的 ``QuadUniformVelocityCommand``。

  共享 ``UniformVelocityCommandCfg.build`` 硬编码返回 ``UniformVelocityCommand``
  且无 ``class_type`` 字段(不可改共享),这里子类覆盖 ``build`` 即可在不动共享的
  前提下给四足换命令实例。采样/更新语义与父类完全一致。
  """

  def build(self, env: ManagerBasedRlEnv) -> QuadUniformVelocityCommand:
    return QuadUniformVelocityCommand(self, env)


def _tilt_angle(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  body_axis: tuple[float, float, float],
) -> torch.Tensor:
  """Return the angle (rad) between ``body_axis`` (world) and world up.

  World-up reference is chosen (no terrain sensor needed, correct on flat
  ground); swap to a terrain normal / sensor branch if quad moves to rough.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # 四足躯干=根 body,直接用 root quat,避开 body_names("base_link")可能有
  # 多个匹配、body_link_quat_w 退化成 [B, N, 4] 的歧义。也省一次索引。
  body_quat_w = asset.data.root_link_quat_w  # [B, 4]
  # expand 到批次维:quat_apply 要求 vec 与 quat 同 batch,单维向量会在
  # B>1 时报 shape 错(共享 upright/debug_vis 同款写法)。
  axis_w = quat_apply(
    body_quat_w,
    torch.tensor(body_axis, device=env.device).expand_as(body_quat_w[:, :3]),
  )
  up = torch.tensor([0.0, 0.0, 1.0], device=env.device)
  cos_theta = (axis_w * up).sum(dim=-1).clamp(-1.0, 1.0)
  return torch.acos(cos_theta)


class quad_upright:
  """四足朝向奖励(带 ``tilt_deg`` 偏角)+ viewer 标定箭头。

  与共享 ``velocity.mdp.rewards.upright`` 同构的 class-based term,但对象
  是"躯干背轴(+Z)与世界 up 夹角 ≈ ``tilt_deg``"。base 前俯 ≈90° 时背轴
  水平、夹角 90 ≈ 目标。绘图在 `debug_vis`:画两根从 base 出发的箭头——
  **橙=当前背轴**、**紫=目标(按 ``tilt_deg`` 转好)方向**,夹角即当前与
  标定目标的偏离。改 ``tilt_deg`` 只动紫色箭头,绝不旋转机器人。
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._debug_vis_enabled = True
    self._asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._tilt_deg = float(cfg.params.get("tilt_deg", QUAD_TILT_DEG))
    self._body_axis = cfg.params.get("body_axis", QUAD_BODY_AXIS)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    tilt_deg: float = QUAD_TILT_DEG,
    body_axis: tuple[float, float, float] = QUAD_BODY_AXIS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    """Reward peak when the quad torso tilt matches ``tilt_deg``.

    ``std`` in rad (shared ``upright`` convention): ``exp(-(θ - tilt_deg)²/std²)``.
    """
    angle = _tilt_angle(env, asset_cfg, body_axis)  # [B] rad
    tilt = torch.tensor(math.radians(tilt_deg), device=env.device)
    return torch.exp(-torch.square(angle - tilt) / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # 状态无关。

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    body_quat_w = asset.data.root_link_quat_w  # [B, 4]
    axis = torch.tensor(self._body_axis, device=env.device)
    axis_w = quat_apply(
      body_quat_w, axis.expand_as(body_quat_w[:, :3])
    )  # [B,3] 当前背轴
    up = torch.tensor([0.0, 0.0, 1.0], device=env.device)

    # 目标背轴:绕 up 抬到 world-up 夹角 = tilt_deg,方位沿用当前背轴水平分量。
    tilt = torch.tensor(math.radians(self._tilt_deg), device=env.device)
    horiz = axis_w - up.unsqueeze(0) * (axis_w * up).sum(dim=-1, keepdim=True)
    horiz_n = torch.norm(horiz, dim=-1, keepdim=True).clamp_min(1e-8)
    target = torch.sin(tilt) * horiz / horiz_n + torch.cos(tilt) * up.unsqueeze(
      0
    ).expand_as(horiz)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    axis_np = axis_w.cpu().numpy()
    target_np = target.cpu().numpy()
    offset = np.array([0.0, 0.4, 0.0])  # 侧向偏移,避免与 base 描线重叠。
    scale = 0.3

    for i in env_indices:
      origin = positions[i] + offset
      # 橙=当前背轴方向;紫=目标(tilt_deg 转过)方向。两箭头夹角=当前与目标偏离。
      visualizer.add_arrow(
        start=origin,
        end=origin + axis_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.9),  # 橙:当前背轴
        width=0.01,
      )
      visualizer.add_arrow(
        start=origin,
        end=origin + target_np[i] * scale,
        color=(0.8, 0.0, 1.0, 0.9),  # 紫:目标背轴(tilt_deg)
        width=0.01,
      )


def quad_bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  tilt_deg: float = QUAD_TILT_DEG,
  body_axis: tuple[float, float, float] = QUAD_BODY_AXIS,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the quad torso tilt deviates from ``tilt_deg`` by > limit.

  ``limit_angle`` in rad (shared ``bad_orientation`` convention). Catches
  side-roll / belly-up (back axis loses "up").
  """
  angle = _tilt_angle(env, asset_cfg, body_axis)  # [B] rad
  tilt = torch.tensor(math.radians(tilt_deg), device=env.device)
  return (angle - tilt).abs() > limit_angle
