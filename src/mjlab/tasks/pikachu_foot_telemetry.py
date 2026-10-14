"""Pikachu 足底状态遥测（play / 录制专用，约 99 列）。

回答"这只脚此刻踩得怎么样"：接触力与冲击、与地面的平行度、有效接触面积、
CoP 与支撑边界、滑移、穿透量、摩擦锥裕度、步态相位与占空比，以及触地瞬间的
姿态与关节冲击负载。

数据来源分两路，各有各的用途：

1. **MuJoCo 接触传感器**（``feet_ground_contact``，``reduce="netforce"``）——
   世界系净接触力。力/冲击/滑移/摩擦锥用它，这是接触求解器的真值。
2. **足底虚拟传感器阵列**（taxel grid）——从模型里读脚掌胶囊 geom
   （``*_foot[1-5]_collision``），在每根胶囊轴上取 ``SENSORS_PER_CAPSULE`` 个
   采样点，沿"站立时朝下"的方向外移一个胶囊半径，得到贴在鞋底的一层 taxel。
   逐帧把它们变换到世界系、与地面平面比高，得到穿透量 / 接触面积 / CoP /
   压力分布。这一路是**几何代理**（鞋底到地面平面的距离），不是接触求解器的
   输出；好处是能给出真实传感器阵列那样的空间分布，而 ``netforce`` 每只脚
   只有一个净力。

两路在 ``foot_pressure_pa`` 上汇合：把**实测**法向力按各 taxel 的穿透量加权
分摊，得到每个 taxel 的压力（Pa）——力的总量是真值，空间分布是几何代理。

坐标约定：CoP 与支撑裕度都在**踝 body 系**里算，这样跨时间/跨步态相位可比，
不受机体平动影响。

⚠️ 地面被当作高度 ``ground_z`` 的水平面（平地任务成立）。斜坡/崎岖地形上
穿透量与接触面积会失真，力和滑移不受影响。

⚠️ 只在 play / 录制时注册（``add_pikachu_foot_telemetry_metrics``）：近 100 列
会往 ``Episode_Metrics/*`` 灌一堆训练时没人看的 tag，且每步多算一次阵列。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast
from weakref import WeakKeyDictionary

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# 脚掌碰撞 geom：Pikachu 每只脚 5 根沿 y 铺开的胶囊（见 v031 训练 XML）。
FOOT_GEOM_PATTERN = r"^(left|right)_foot\d+_collision$"
ANKLE_BODY_PATTERN = r"^(left|right)_ankle_link$"
# 每根胶囊上取几个采样点（沿轴向等距）。5 根 × 5 点 = 每脚 25 个 taxel，
# 排成 5 行(y，脚掌横向) × 5 列(x，脚掌纵向) 的"传感器垫"。
SENSORS_PER_CAPSULE = 5
# Pikachu 每只脚的胶囊根数。列名数量在配置期就要定下来，所以写成常量并在建
# 阵列时断言模型确实有这么多。
FOOT_CAPSULE_COUNT = 5
N_TAXELS_PER_FOOT = FOOT_CAPSULE_COUNT * SENSORS_PER_CAPSULE
# 鞋底弹性垫厚度（m）。**这是本模块唯一一个建模假设**：刚体脚踩刚性地面的几何
# 接触是退化的一条线/一个点，直接按"taxel 陷进地面"判接触，接触面积恒≈0。
# 真实鞋垫有厚度，压痕随载荷与倾角连续变化。这里把鞋底当作一层厚
# SOLE_COMPLIANCE_M 的弹性垫：间隙小于垫厚就算压上了，压痕 = 垫厚 - 间隙。
# 面积/CoP/压力分布都建立在压痕上（力的大小仍用接触传感器的实测值）。
SOLE_COMPLIANCE_M = 0.002

FEET_CONTACT_SENSOR = "feet_ground_contact"
FOOT_SITE_NAMES = ("left_foot", "right_foot")

_SIDES = ("left", "right")
# 少于这么长的"支撑+腾空"不当作一个完整步态周期：play 刚起步时 last_*_time 还
# 是初值，拿它算相位/占空比会得出毫无意义的数（宁可报 NaN）。
_MIN_CYCLE_S = 0.05
# 支撑多边形裕度用的方向采样数：CoP 到接触凸包边界的距离取
# ``min_u max_i (p_i - c)·u``，u 均匀取这么多方向。凸多边形上该值在边法向处
# 取到真值，采样 32 向的误差在毫米量级，判读"CoP 离边界多远"足够。
_HULL_DIRECTIONS = 32


def _quat_to_mat_np(q: np.ndarray) -> np.ndarray:
  """wxyz 四元数 → 3x3 旋转矩阵（numpy，仅建阵列时用一次）。"""
  w, x, y, z = (float(v) for v in q)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


@dataclass(frozen=True)
class FootSensorGrid:
  """足底虚拟传感器阵列的静态几何（建一次，逐帧复用）。"""

  local_pos: torch.Tensor
  """(S, 3) 每个 taxel 在所属踝 body 系下的位置，按 (脚, y, x) 排序。"""
  local_normal: torch.Tensor
  """(S, 3) 每个 taxel 朝外的法向（踝 body 系）。"""
  side: torch.Tensor
  """(S,) 0=左 1=右，与 ``local_pos`` 同序。"""
  body_local_ids: torch.Tensor
  """(2,) 左右踝 body 在 ``entity.body_names`` 里的局部索引。"""
  cell_area_m2: float
  """单个 taxel 代表的鞋底面积（m²）：鞋底投影面积 / taxel 数。"""
  ground_z: float
  """地面平面高度（m）。"""
  n_per_foot: int
  """每只脚的 taxel 数。"""


def _ground_height(model: mujoco.MjModel) -> float:
  """取地形平面 geom 的高度；找不到平面就退回 0.0（平地任务的地面在 z=0）。"""
  plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
  for g in range(model.ngeom):
    if int(model.geom_type[g]) == plane:
      return float(model.geom_pos[g][2])
  return 0.0


def build_foot_sensor_grid(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
  per_capsule: int = SENSORS_PER_CAPSULE,
) -> FootSensorGrid:
  """从编译好的模型里推出足底 taxel 阵列。

  每根脚掌胶囊贡献 ``per_capsule`` 个点：沿胶囊轴等距取点，再沿"站立时朝下"
  的方向外移一个胶囊半径，落到鞋底表面上。朝向由**模型参考位姿**（``qpos0``）
  下该 body 的世界姿态反推，所以左右脚各自镜像的坐标系自动正确，不写死 -z。
  """
  entity: Entity = env.scene[entity_name]
  model = env.sim.mj_model

  local_ids, names = entity.find_geoms(FOOT_GEOM_PATTERN, preserve_order=True)
  if not local_ids:
    msg = (
      f"{entity_name} 上没有匹配 {FOOT_GEOM_PATTERN} 的脚掌 geom，"
      "无法建足底传感器阵列（检查资产命名）"
    )
    raise ValueError(msg)
  global_ids = [int(entity.indexing.geom_ids[i]) for i in local_ids]

  # 参考位姿：qpos = qpos0 时各 body 的世界姿态。用它把"世界朝下"映射回 body 系。
  rest = mujoco.MjData(model)
  rest.qpos[:] = model.qpos0
  mujoco.mj_forward(model, rest)

  positions: list[np.ndarray] = []
  normals: list[np.ndarray] = []
  sides: list[int] = []
  radius_sum = 0.0
  for gid, name in zip(global_ids, names, strict=True):
    if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
      msg = f"脚掌 geom {name} 不是 capsule（type={int(model.geom_type[gid])}）"
      raise ValueError(msg)
    gpos = np.asarray(model.geom_pos[gid], dtype=np.float64)
    gquat = np.asarray(model.geom_quat[gid], dtype=np.float64)
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    radius, half_len = float(size[0]), float(size[1])
    axis = _quat_to_mat_np(gquat) @ np.array([0.0, 0.0, 1.0])  # 胶囊轴（body 系）

    body_gid = int(model.geom_bodyid[gid])
    # rest.xmat 是展平的 3x3（9 元素），不是四元数。
    rest_mat = np.asarray(rest.xmat[body_gid], dtype=np.float64).reshape(3, 3)
    down = rest_mat.T @ np.array([0.0, 0.0, -1.0])
    # 鞋底朝外方向 = "朝下"在胶囊横截面上的投影（去掉沿轴分量）。
    normal = down - axis * float(down @ axis)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
      msg = f"脚掌 geom {name} 的胶囊轴与站立朝下方向平行，无法定出鞋底法向"
      raise ValueError(msg)
    normal = normal / norm

    for s in np.linspace(-half_len, half_len, per_capsule):
      positions.append(gpos + axis * float(s) + normal * radius)
      normals.append(normal)
      sides.append(0 if name.startswith("left") else 1)
    radius_sum += radius

  pos = np.asarray(positions, dtype=np.float64)
  side_arr = np.asarray(sides, dtype=np.int64)
  # 排序：先按脚，再按 y（行）、x（列），让压力热力图的轴是稳定的物理布局。
  order = np.lexsort((pos[:, 0], pos[:, 1], side_arr))
  pos, side_arr = pos[order], side_arr[order]
  nrm = np.asarray(normals, dtype=np.float64)[order]

  n_per_foot = int(np.sum(side_arr == 0))
  if n_per_foot * 2 != pos.shape[0]:
    msg = f"左右脚 taxel 数不等（{n_per_foot} vs {pos.shape[0] - n_per_foot}）"
    raise ValueError(msg)
  radius_mean = radius_sum / len(global_ids)
  x_span = float(pos[:, 0].max() - pos[:, 0].min())
  y_span = float(pos[:, 1].max() - pos[:, 1].min())
  sole_area = (x_span + 2 * radius_mean) * (y_span + 2 * radius_mean)

  body_local_ids, _ = entity.find_bodies(ANKLE_BODY_PATTERN, preserve_order=True)
  if len(body_local_ids) != 2:
    msg = f"期望 2 个踝 body，实得 {len(body_local_ids)} 个（{ANKLE_BODY_PATTERN}）"
    raise ValueError(msg)

  device = env.device
  return FootSensorGrid(
    local_pos=torch.as_tensor(pos, dtype=torch.float32, device=device),
    local_normal=torch.as_tensor(nrm, dtype=torch.float32, device=device),
    side=torch.as_tensor(side_arr, dtype=torch.long, device=device),
    body_local_ids=torch.as_tensor(body_local_ids, dtype=torch.long, device=device),
    cell_area_m2=sole_area / float(n_per_foot),
    ground_z=_ground_height(model),
    n_per_foot=n_per_foot,
  )


def _side_metrics(stem: str) -> list[str]:
  return [f"{stem}/{side}" for side in _SIDES]


def _pressure_metrics() -> list[str]:
  return [
    f"foot_pressure_pa/{side}_foot{i}"
    for side in _SIDES
    for i in range(N_TAXELS_PER_FOOT)
  ]


#: 每列的 episode 汇总口径。逐帧 CSV 拿到的是当帧真实值，与 reduce 无关；
#: reduce 只决定 tfevents 里 ``Episode_Metrics/*`` 那一行怎么聚合。
_ReduceName = Literal["last", "max", "max_abs", "mean", "sum"]
_REDUCE: dict[str, _ReduceName] = {}
for _stem, _reduce in (
  ("foot_contact", "mean"),
  ("foot_phase", "mean"),
  ("foot_duty", "mean"),
  ("foot_stance_time_s", "mean"),
  ("foot_air_time_s", "mean"),
  ("foot_force_normal_n", "max"),
  ("foot_force_tangent_n", "max"),
  ("foot_force_rate_n_s", "max"),
  ("foot_impact_peak_n", "max"),
  ("foot_impact_rise_n_s", "max"),
  ("foot_contact_slip_speed_m_s", "max"),
  ("foot_contact_slip_dist_m", "max"),
  ("foot_slip_speed_peak_m_s", "max"),
  ("foot_penetration_m", "max"),
  ("foot_contact_area_m2", "mean"),
  ("foot_contact_fraction", "mean"),
  ("foot_sensor_count", "mean"),
  ("foot_tilt_deg", "max"),
  ("foot_pitch_deg", "max_abs"),
  ("foot_roll_deg", "max_abs"),
  ("foot_touchdown_pitch_deg", "last"),
  ("foot_touchdown_roll_deg", "last"),
  ("foot_touchdown_tilt_deg", "last"),
  ("foot_cop_x_m", "mean"),
  ("foot_cop_y_m", "mean"),
  ("foot_cop_margin_m", "mean"),
  ("foot_friction_margin_n", "mean"),
  ("foot_friction_util", "max"),
):
  for _name in _side_metrics(_stem):
    _REDUCE[_name] = cast("_ReduceName", _reduce)
for _name in _pressure_metrics():
  _REDUCE[_name] = "max"
_REDUCE["foot_ground_friction_mu"] = "mean"
_REDUCE["base_roll_deg"] = "max_abs"
_REDUCE["base_roll_rate_rad_s"] = "max_abs"
del _stem, _reduce, _name


def joint_torque_rate_names(joint_names: tuple[str, ...]) -> list[str]:
  """触地冲击负载的列名：逐驱动关节的 |dτ/dt|。"""
  return [f"joint_torque_rate_nm_s/{j}" for j in joint_names]


@dataclass(frozen=True)
class FootVisualState:
  """viser 3D 叠加层用的几何快照（**全批次**，调用方按 ``env_idx`` 索引）。

  张量形状（B=env 数，S=50 个 taxel，N=25 taxel/脚）：

    taxel_w [B,S,3]      taxel 世界坐标        in_contact  [B,S]    bool
    pressure [B,2,N]     taxel 压力 (Pa)       penetration [B,S]    穿透 (m)
    cop_w [B,2,3]        CoP 世界坐标          has_contact [B,2,1]  有接触
    sole_up [B,2,3]      鞋底朝上法向          force_w [B,2,3]      世界系净力
    force_n [B,2]        法向力 (N)            force_t [B,2]        切向力 (N)
    mu [B]               足部摩擦系数          body_pos [B,2,3]     踝 link 世界位置
    hull_local [2,N,2]   踝体系 taxel xy（支撑多边形用，与批次无关）
  """

  step: int
  taxel_w: torch.Tensor
  pressure: torch.Tensor
  in_contact: torch.Tensor
  cop_w: torch.Tensor
  has_contact: torch.Tensor
  sole_up: torch.Tensor
  body_pos: torch.Tensor
  force_w: torch.Tensor
  force_n: torch.Tensor
  force_t: torch.Tensor
  mu: torch.Tensor
  penetration: torch.Tensor
  hull_local: torch.Tensor
  cell_area_m2: float
  ground_z: float


class FootTelemetry:
  """足底遥测的运行时状态与逐步缓存。

  一次 :meth:`values` 调用算完所有列；同一控制步内重复调用直接命中缓存，
  所以近 100 个 metric term 与 viser 叠加层共享一次计算、跨步状态也只推进一次。
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    sensor_name: str = FEET_CONTACT_SENSOR,
    joint_names: tuple[str, ...] = (),
  ) -> None:
    self.entity_name = entity_name
    self.sensor_name = sensor_name
    self.joint_names = tuple(joint_names)
    self.grid = build_foot_sensor_grid(env, entity_name)
    if self.grid.n_per_foot != N_TAXELS_PER_FOOT:
      msg = (
        f"每脚 taxel 数 {self.grid.n_per_foot} 与列名约定的 {N_TAXELS_PER_FOOT} 不符"
      )
      raise ValueError(msg)

    entity: Entity = env.scene[entity_name]
    device = env.device
    n = env.num_envs

    self._cache_step = -1
    self._cache: dict[str, torch.Tensor] = {}
    self._vis: FootVisualState | None = None

    # 逐 env 的跨步状态。
    self._prev_contact = torch.zeros(n, 2, device=device)
    self._prev_force_n = torch.zeros(n, 2, device=device)
    self._prev_site_xy = torch.zeros(n, 2, 2, device=device)
    self._prev_joint_torque = torch.zeros(n, len(self.joint_names), device=device)
    self._slip_dist = torch.zeros(n, 2, device=device)
    self._slip_peak = torch.zeros(n, 2, device=device)
    self._impact_peak = torch.zeros(n, 2, device=device)
    self._impact_rise = torch.zeros(n, 2, device=device)
    self._td_pitch = torch.zeros(n, 2, device=device)
    self._td_roll = torch.zeros(n, 2, device=device)
    self._td_tilt = torch.zeros(n, 2, device=device)

    # 支撑多边形裕度的方向采样（[U,2]）。
    ang = torch.arange(_HULL_DIRECTIONS, device=device) * (
      2.0 * torch.pi / _HULL_DIRECTIONS
    )
    self._hull_dirs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    # 脚掌 geom 的全局 id（读摩擦系数用），以及左右分组掩码。
    local_ids, _ = entity.find_geoms(FOOT_GEOM_PATTERN, preserve_order=True)
    self._foot_geom_ids = entity.indexing.geom_ids[local_ids]
    self._side_mask = [self.grid.side == s for s in range(2)]
    # 每脚 taxel 的局部坐标（[2,N,3]）。支撑裕度用它的 xy；CoP 也在这里加权，
    # 于是 CoP 是接触 taxel 局部坐标的凸组合，不必绕世界系再逆旋转回来。
    local_xyz = self.grid.local_pos
    self._local_pos_by_side = torch.stack(
      [local_xyz[self._side_mask[s]] for s in range(2)], dim=0
    )
    self._local_xy_by_side = self._local_pos_by_side[..., :2]

    self._site_ids = entity.find_sites(FOOT_SITE_NAMES, preserve_order=True)[0]
    self._joint_ids = [
      entity.joint_names.index(j) for j in self.joint_names if j in entity.joint_names
    ]
    # 传感器 primary 槽位 → 左右脚的映射（primary 顺序由正则展开决定，别假设）。
    self._sensor_perm = self._resolve_sensor_perm(env)

  def _resolve_sensor_perm(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    sensor: ContactSensor = env.scene[self.sensor_name]
    names = sensor.primary_names
    perm = []
    for side in _SIDES:
      idx = next((i for i, nm in enumerate(names) if side in nm), None)
      if idx is None:
        msg = f"接触传感器 {self.sensor_name} 的 primary {names} 里找不到 {side}"
        raise ValueError(msg)
      perm.append(idx)
    return torch.as_tensor(perm, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """清空指定 env 的跨步状态（滑移累积、冲击峰值、触地锁存）。"""
    for buf in (
      self._prev_contact,
      self._prev_force_n,
      self._prev_site_xy,
      self._prev_joint_torque,
      self._slip_dist,
      self._slip_peak,
      self._impact_peak,
      self._impact_rise,
      self._td_pitch,
      self._td_roll,
      self._td_tilt,
    ):
      buf[env_ids] = 0.0

  def values(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """当前控制步的全部足底列（命中缓存时不重算、不推进状态）。"""
    if env.common_step_counter == self._cache_step:
      return self._cache
    self._cache_step = env.common_step_counter
    self._cache = self._compute(env)
    return self._cache

  def visual_state(self, env: ManagerBasedRlEnv) -> FootVisualState | None:
    """3D 叠加层用的几何快照（全批次，调用方自行按 ``env_idx`` 索引）。

    与 :meth:`values` 命中同一份缓存，所以 viser 每帧取一次不会重算，也不会
    二次推进滑移/冲击等跨步状态。返回 ``None`` 表示本步还没算过。
    """
    if env.common_step_counter != self._cache_step:
      self.values(env)
    return self._vis

  # 计算。

  def _compute(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    grid = self.grid
    entity: Entity = env.scene[self.entity_name]
    sensor: ContactSensor = env.scene[self.sensor_name]
    device = env.device
    n = env.num_envs
    step_dt = max(float(env.step_dt), 1e-6)

    # 刚 reset 的 env：episode_length_buf 在这一步是 0（reset 内清零）或 1
    # （reset 后第一次 compute）。两种情况的跨步状态都该从零开始。
    fresh = env.episode_length_buf <= 1
    if bool(fresh.any()):
      self.reset(fresh)

    # --- 接触传感器：世界系净力 [B,2,3]，已按左右重排。
    assert sensor.data.found is not None and sensor.data.force is not None
    perm = self._sensor_perm
    found = sensor.data.found[:, perm].float()
    force_w = sensor.data.force[:, perm, :].float()
    contact = (found > 0).float()
    # MuJoCo 的接触力方向是 primary→secondary（脚 → 地面），所以地面给脚的支持力
    # 是 -z。取反后左右之和 ≈ mg（实测 65.8 N vs 66.4 N），可直接当法向支持力用。
    force_n = torch.clamp(-force_w[..., 2], min=0.0)
    force_t = torch.norm(force_w[..., :2], dim=-1)
    zeros = torch.zeros_like(contact)
    nan_like = torch.full_like(contact, float("nan"))
    contact_time = (
      zeros
      if sensor.data.current_contact_time is None
      else sensor.data.current_contact_time[:, perm]
    )
    air_time = (
      zeros
      if sensor.data.current_air_time is None
      else sensor.data.current_air_time[:, perm]
    )
    last_contact = (
      zeros
      if sensor.data.last_contact_time is None
      else sensor.data.last_contact_time[:, perm]
    )
    last_air = (
      torch.ones_like(contact)
      if sensor.data.last_air_time is None
      else sensor.data.last_air_time[:, perm]
    )

    # 步态相位：0 = 刚触地，1 = 该周期结束。周期长度取"上一次完整支撑+腾空"。
    # 第一个完整周期走完之前没有可信的周期长度，相位/占空比报 NaN 而不是编一个数。
    cycle = torch.clamp(last_contact + last_air, min=1e-3)
    have_cycle = (last_contact + last_air) > _MIN_CYCLE_S
    duty = torch.where(
      have_cycle, torch.clamp(last_contact / cycle, 0.0, 1.0), nan_like
    )
    phase = torch.where(
      have_cycle,
      torch.remainder(
        torch.where(
          contact > 0.5, contact_time / cycle, (last_contact + air_time) / cycle
        ),
        1.0,
      ),
      nan_like,
    )

    # --- taxel 阵列：世界位置、穿透量、法向。
    body_pose = entity.data.body_link_pose_w[:, grid.body_local_ids, :]  # [B,2,7]
    body_pos = body_pose[..., 0:3]  # [B,2,3]
    body_quat = body_pose[..., 3:7]  # [B,2,4] wxyz

    taxel_body = body_pos[:, grid.side, :]  # [B,S,3]
    taxel_quat = body_quat[:, grid.side, :]  # [B,S,4]
    local_pos = grid.local_pos.unsqueeze(0).expand(n, -1, -1)
    taxel_w = taxel_body + quat_apply(taxel_quat, local_pos)  # [B,S,3]
    normal_w = quat_apply(taxel_quat, grid.local_normal.unsqueeze(0).expand(n, -1, -1))

    gap = taxel_w[..., 2] - grid.ground_z  # [B,S] >0 表示悬空
    penetration = -gap  # >0 表示陷进地面
    # 弹性垫模型：间隙小于垫厚就算压上了，压痕随载荷/倾角连续变化（见 SOLE_COMPLIANCE_M）。
    compression = (SOLE_COMPLIANCE_M - gap).clamp(min=0.0)
    in_contact = compression > 0.0

    # 逐脚聚合：把 [B,S,...] 拆成 [B,2,N,...]。
    def _by_side(t: torch.Tensor) -> torch.Tensor:
      return torch.stack([t[:, m] for m in self._side_mask], dim=1)

    pen_f = _by_side(penetration)  # [B,2,N]
    weight = _by_side(compression)  # [B,2,N]，非接触点为 0，可直接当权重
    contact_f = _by_side(in_contact.float())  # [B,2,N]
    normal_f = _by_side(normal_w)  # [B,2,N,3]

    sensor_count = contact_f.sum(dim=-1)  # [B,2]
    contact_area = sensor_count * grid.cell_area_m2
    contact_fraction = sensor_count / float(grid.n_per_foot)

    # 鞋底朝向：taxel 法向均值 → 与地面法向（世界 z）的夹角，及 pitch/roll 分量。
    # 阵列存的是鞋底**朝外**（站立时朝下）的法向，取反才是指向地面的那一侧，
    # 也就是"鞋底面"的向上法向 —— 与地面的夹角、pitch、roll 都按它算，
    # 于是水平踩实 = 0°，而不是 180°。
    sole_up = torch.nn.functional.normalize(-normal_f.mean(dim=-2), dim=-1)
    tilt = torch.rad2deg(torch.acos(torch.clamp(sole_up[..., 2], -1.0, 1.0)))
    pitch = torch.rad2deg(torch.asin(torch.clamp(sole_up[..., 0], -1.0, 1.0)))
    roll = -torch.rad2deg(torch.asin(torch.clamp(sole_up[..., 1], -1.0, 1.0)))

    # CoP：按压痕加权（非接触点权重为 0）。
    # 在**踝体系**里对 taxel 局部坐标直接加权平均，而不是"世界系加权后逆旋转"：
    # 后者得先给 cop_w 补一个 z，补 0 等于把 CoP 当成落在地面高度上，与真实位置
    # 差一个穿透量（实测最大 5 mm）；脚一倾斜这个 z 误差就经旋转泄漏到 xy
    # （tilt 28° 时约 0.77 mm），把 CoP 推出接触 taxel 的凸包，支撑裕度出现假负值。
    # 局部加权平均是凸组合，margin ≥ 0 有保证。
    weight_sum = weight.sum(dim=-1, keepdim=True)
    has_contact = weight_sum > 1e-9
    local_f = self._local_pos_by_side.unsqueeze(0)  # [1,2,N,3]
    cop_local_raw = (local_f * weight.unsqueeze(-1)).sum(dim=-2) / weight_sum.clamp(
      min=1e-9
    )  # [B,2,3] 踝体系
    cop_w = body_pos + quat_apply(body_quat, cop_local_raw)  # [B,2,3] 世界系
    # 没接触就没有 CoP，报 NaN 而不是让它落在 (0,0) 相对踝的位置上。
    no_cop = ~has_contact
    nan_col = nan_like.unsqueeze(-1)
    cop_local = torch.where(no_cop, nan_col.expand_as(cop_local_raw), cop_local_raw)
    cop_w = torch.where(no_cop, nan_col.expand_as(cop_w), cop_w)

    # 支撑边界裕度：min_u max_i (p_i - c)·u，p_i 用踝体系下的 taxel 位置。
    rel = self._local_xy_by_side.unsqueeze(0) - cop_local[..., None, :2]  # [B,2,N,2]
    proj = torch.where(
      contact_f.unsqueeze(-1) > 0.5, rel @ self._hull_dirs.T, -1e6
    )  # [B,2,N,U]
    margin = proj.max(dim=-2).values.min(dim=-1).values  # [B,2]
    margin = torch.where(
      sensor_count > 0.5, margin, torch.full_like(margin, float("nan"))
    )

    # 压力：把实测法向力按压痕分摊到各 taxel（力的总量是真值，分布是几何代理）。
    share = weight / weight_sum.clamp(min=1e-9)  # [B,2,N]
    pressure = share * force_n.unsqueeze(-1) / grid.cell_area_m2  # [B,2,N]

    # --- 滑移：接触时的水平位移与速度（触地那一帧的位移发生在腾空段，不计）。
    site_xy = entity.data.site_pos_w[:, self._site_ids, :2]  # [B,2,2]
    foot_speed = torch.norm(entity.data.site_lin_vel_w[:, self._site_ids, :2], dim=-1)
    rising = (contact > 0.5) & (self._prev_contact <= 0.5)
    step_dist = torch.norm(site_xy - self._prev_site_xy, dim=-1)  # [B,2]
    self._slip_dist = torch.where(rising, zeros, self._slip_dist)
    self._slip_dist = self._slip_dist + step_dist * contact * (~rising).float()
    slip_speed = foot_speed * contact
    self._slip_peak = torch.where(rising, zeros, self._slip_peak)
    self._slip_peak = torch.maximum(self._slip_peak, slip_speed)

    # --- 冲击：法向力变化率，及其在本次支撑内的峰值。
    force_rate = (force_n - self._prev_force_n) / step_dt
    self._impact_peak = torch.where(rising, zeros, self._impact_peak)
    self._impact_rise = torch.where(rising, zeros, self._impact_rise)
    self._impact_peak = torch.maximum(self._impact_peak, force_n * contact)
    self._impact_rise = torch.maximum(
      self._impact_rise, force_rate.clamp(min=0.0) * contact
    )

    # --- 触地瞬间姿态锁存。
    self._td_pitch = torch.where(rising, pitch, self._td_pitch)
    self._td_roll = torch.where(rising, roll, self._td_roll)
    self._td_tilt = torch.where(rising, tilt, self._td_tilt)

    # --- 摩擦锥：μ 取该 env 当前足部 geom 的实际摩擦系数（启动 DR 会改它）。
    mu = env.sim.model.geom_friction[:, self._foot_geom_ids, 0].mean(dim=-1)  # [B]
    friction_margin = mu.unsqueeze(-1) * force_n - force_t
    friction_util = torch.where(
      force_n > 1e-3,
      force_t / (mu.unsqueeze(-1) * force_n).clamp(min=1e-6),
      torch.zeros_like(force_t),
    )

    # --- 机体横滚角与横滚角速度。
    # 约定与 foot_roll_deg 一致：正值 = 绕世界 +x 轴右手正转 = 左侧(+y)抬起、
    # 向右倾。projected_gravity_b = R^T @ (0,0,-1)，展开后 g_y = -sinφ·cosθ、
    # g_z = -cosφ·cosθ，故 φ = atan2(-g_y, -g_z)。用 asin(g_y) 会整体反号
    # （标定实测：根横滚 +10° 时 asin 给 -10.02°，atan2 给 +10.0°）。
    gravity_b = entity.data.projected_gravity_b  # [B,3]
    base_roll = torch.rad2deg(torch.atan2(-gravity_b[:, 1], -gravity_b[:, 2]))
    base_roll_rate = entity.data.root_link_ang_vel_b[:, 0]

    # --- 关节冲击负载：|dτ/dt|。
    if self._joint_ids:
      torque = entity.data.qfrc_actuator[:, self._joint_ids]
      torque_rate = torch.abs(torque - self._prev_joint_torque) / step_dt
    else:
      torque = torch.zeros(n, 0, device=device)
      torque_rate = torch.zeros(n, 0, device=device)

    out: dict[str, torch.Tensor] = {}
    for i, side in enumerate(_SIDES):
      out[f"foot_contact/{side}"] = contact[:, i]
      out[f"foot_phase/{side}"] = phase[:, i]
      out[f"foot_duty/{side}"] = duty[:, i]
      out[f"foot_stance_time_s/{side}"] = contact_time[:, i]
      out[f"foot_air_time_s/{side}"] = air_time[:, i]
      out[f"foot_force_normal_n/{side}"] = force_n[:, i]
      out[f"foot_force_tangent_n/{side}"] = force_t[:, i]
      out[f"foot_force_rate_n_s/{side}"] = force_rate[:, i]
      out[f"foot_impact_peak_n/{side}"] = self._impact_peak[:, i]
      out[f"foot_impact_rise_n_s/{side}"] = self._impact_rise[:, i]
      out[f"foot_contact_slip_speed_m_s/{side}"] = slip_speed[:, i]
      out[f"foot_contact_slip_dist_m/{side}"] = self._slip_dist[:, i]
      out[f"foot_slip_speed_peak_m_s/{side}"] = self._slip_peak[:, i]
      out[f"foot_penetration_m/{side}"] = pen_f[:, i].clamp(min=0.0).max(dim=-1).values
      out[f"foot_contact_area_m2/{side}"] = contact_area[:, i]
      out[f"foot_contact_fraction/{side}"] = contact_fraction[:, i]
      out[f"foot_sensor_count/{side}"] = sensor_count[:, i]
      out[f"foot_tilt_deg/{side}"] = tilt[:, i]
      out[f"foot_pitch_deg/{side}"] = pitch[:, i]
      out[f"foot_roll_deg/{side}"] = roll[:, i]
      out[f"foot_touchdown_pitch_deg/{side}"] = self._td_pitch[:, i]
      out[f"foot_touchdown_roll_deg/{side}"] = self._td_roll[:, i]
      out[f"foot_touchdown_tilt_deg/{side}"] = self._td_tilt[:, i]
      out[f"foot_cop_x_m/{side}"] = cop_local[:, i, 0]
      out[f"foot_cop_y_m/{side}"] = cop_local[:, i, 1]
      out[f"foot_cop_margin_m/{side}"] = margin[:, i]
      out[f"foot_friction_margin_n/{side}"] = friction_margin[:, i]
      out[f"foot_friction_util/{side}"] = friction_util[:, i]
      for k in range(grid.n_per_foot):
        out[f"foot_pressure_pa/{side}_foot{k}"] = pressure[:, i, k]
    out["foot_ground_friction_mu"] = mu
    out["base_roll_deg"] = base_roll
    out["base_roll_rate_rad_s"] = base_roll_rate
    for k, name in enumerate(self.joint_names):
      out[f"joint_torque_rate_nm_s/{name}"] = torque_rate[:, k]

    # 3D 叠加层快照（metric 列里只有标量，画图元要几何量）。全部 detach，不建图。
    self._vis = FootVisualState(
      step=int(env.common_step_counter),
      taxel_w=taxel_w.detach(),
      pressure=pressure.detach(),
      in_contact=in_contact.detach(),
      cop_w=cop_w.detach(),
      has_contact=has_contact.detach(),
      sole_up=sole_up.detach(),
      body_pos=body_pos.detach(),
      force_w=force_w.detach(),
      force_n=force_n.detach(),
      force_t=force_t.detach(),
      mu=mu.detach(),
      penetration=penetration.detach(),
      hull_local=self._local_xy_by_side.detach(),
      cell_area_m2=float(grid.cell_area_m2),
      ground_z=float(grid.ground_z),
    )

    # 状态推进（放最后：上面读到的都是"上一步"的值）。
    self._prev_contact = contact.detach()
    self._prev_force_n = force_n.detach()
    self._prev_site_xy = site_xy.detach()
    if self._joint_ids:
      self._prev_joint_torque = torque.detach().clone()
    return out


_TELEMETRY: WeakKeyDictionary = WeakKeyDictionary()


def foot_telemetry(
  env: ManagerBasedRlEnv,
  joint_names: tuple[str, ...] = (),
  entity_name: str = "robot",
  sensor_name: str = FEET_CONTACT_SENSOR,
) -> FootTelemetry:
  """取（必要时创建）某个 env 的足底遥测对象。"""
  tele = _TELEMETRY.get(env)
  if tele is None:
    tele = FootTelemetry(env, entity_name, sensor_name, joint_names)
    _TELEMETRY[env] = tele
  return tele


def foot_telemetry_registered(env: ManagerBasedRlEnv) -> FootTelemetry | None:
  """已注册的足底遥测对象；没注册过就返回 ``None``（**不创建**）。

  viser 叠加层用它复用 metric 侧建好的那一份。若两边各调一次
  :func:`foot_telemetry`，两个实例会各自推进滑移/冲击等跨步状态，而且谁先
  创建决定了 ``joint_names``，后创建的一方拿不到 ``joint_torque_rate_*`` 列。
  """
  return _TELEMETRY.get(env)


def foot_telemetry_metric_names(joint_names: tuple[str, ...] = ()) -> list[str]:
  """本模块注册的全部列名（顺序 = 注册顺序）。"""
  return list(_REDUCE) + joint_torque_rate_names(joint_names)


def _make_term(
  name: str,
  joint_names: tuple[str, ...],
  entity_name: str,
  sensor_name: str,
):
  """给一列造一个 metric 函数；同一 cfg 的所有列共享同一个遥测对象。"""

  def _term(env: ManagerBasedRlEnv, _name: str = name) -> torch.Tensor:
    tele = foot_telemetry(env, joint_names, entity_name, sensor_name)
    return tele.values(env)[_name]

  _term.__name__ = f"foot_telemetry_{name.replace('/', '_')}"
  return _term


def add_pikachu_foot_telemetry_metrics(
  cfg: ManagerBasedRlEnvCfg,
  joint_names: tuple[str, ...] = (),
  entity_name: str = "robot",
  sensor_name: str = FEET_CONTACT_SENSOR,
) -> None:
  """把足底遥测注册成 metric（逐帧进 CSV、episode 汇总进 tfevents）。

  只该在 play / 录制档调用：约 99 列，训练时注册等于给 ``Episode_Metrics/*``
  灌一整套没人看的 tag。

  ``joint_names`` 决定 ``joint_torque_rate_nm_s/*`` 的列数（触地冲击负载）。
  """
  if not any(s.name == sensor_name for s in cfg.scene.sensors or ()):
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
      ContactSensorCfg(
        name=sensor_name,
        primary=ContactMatch(
          mode="subtree", pattern=ANKLE_BODY_PATTERN, entity=entity_name
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      ),
    )

  for name in foot_telemetry_metric_names(joint_names):
    cfg.metrics[name] = MetricsTermCfg(
      func=_make_term(name, joint_names, entity_name, sensor_name),
      reduce=_REDUCE.get(name, "mean"),
    )
