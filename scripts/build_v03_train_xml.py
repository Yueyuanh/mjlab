"""把 pikachu_v03_base_series 的展示用 MJCF 转成 mjlab 训练用 MJCF。

源模型是 SolidWorks -> easymjcf 转出来的「展示/调试」模型：底座焊死在世界、
几何体没有名字、没有足部 site。mjlab 的训练任务需要：

  1. base_link 上挂 ``<freejoint name="floating_base"/>``（否则机器人不能移动）
  2. geom 显式命名（``*_visual`` / ``*_collision``），mjlab 的 CollisionCfg 用正则匹配
  3. 足部 site（``left_foot`` / ``right_foot``），velocity 任务的 foot_clearance /
     foot_slip 奖励和足底高度扫描传感器都依赖它
  4. IMU site（``imu``），观测里 base 角速度/线速度的参考系
  5. 移除 ``<include file="scene.xml"/>``：mjlab 的地面由 TerrainEntity 提供，
     保留资产自带的 floor 会与地形重叠产生双重地面
  6. 补一个 ``<sensor>`` 块：velocity 任务的 ``base_lin_vel`` / ``base_ang_vel``
     观测直接读 ``robot/imu_lin_vel``（velocimeter）与 ``robot/imu_ang_vel``
     （gyro），``angular_momentum`` 奖励读 ``robot/root_angmom``。源模型没有
     这些传感器（14dof 资产是靠 ``include sensors.xml`` 提供的），缺了会直接
     ``KeyError: Scene element 'robot/imu_lin_vel' not found``。

碰撞策略：平地行走训练用 FEET_ONLY_COLLISION（只有脚底与地面接触），
所以这里只给 ankle 生成脚底胶囊，其余 link 一律只做视觉。
这既避开 mesh-mesh 自碰撞的开销（base_link 的 STL 有近 10 MB 三角形），
也符合 mjlab 现有 14dof 资产的约定。

用法：
    uv run python scripts/build_v03_train_xml.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import mujoco
import numpy as np

SRC_DIR = Path("/home/finnox/Pikachu/PikachuRobot/pikachu_urdf/pikachu_v03_base_series")
SRC_XML = SRC_DIR / "mjcf" / "pikachu_v03_base_series.xml"
SRC_MESHES = SRC_DIR / "meshes"

DST_DIR = (
  Path(__file__).resolve().parent.parent
  / "src"
  / "mjlab"
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "mjcf_v03"
)
DST_XML = DST_DIR / "pikachu_v03_base_series_train.xml"
DST_MESHES = DST_DIR / "meshes"

# 脚底胶囊参数：由 ankle mesh 在 body 系下的 AABB 实测得到。
# x ±0.0465（脚长 9.3 cm）、y ±0.0355（脚宽 7.1 cm）、鞋底 z=-0.034。
FOOT_SOLE_Z = -0.034
FOOT_RADIUS = 0.012
# 胶囊的 fromto 是【轴线】，碰撞面在轴线外 FOOT_RADIUS 处。
# 要让胶囊底面正好落在 mesh 鞋底平面上，轴线必须抬高一个半径。
FOOT_AXIS_Z = FOOT_SOLE_Z + FOOT_RADIUS
FOOT_X_HALF = 0.040
FOOT_Y_OFFSETS = (-0.028, -0.014, 0.0, 0.014, 0.028)


def ankle_aabb() -> tuple[np.ndarray, np.ndarray]:
  """从源模型读出 ankle mesh 在 body 系下的 AABB（已应用 geom 的 pos/quat）。"""
  m = mujoco.MjModel.from_xml_path(str(SRC_XML))
  best: tuple[np.ndarray, np.ndarray] | None = None
  for b in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if not name or "ankle" not in name:
      continue
    for g in range(m.ngeom):
      if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
      did = m.geom_dataid[g]
      va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
      v = m.mesh_vert[va : va + vn].reshape(-1, 3)
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, m.geom_quat[g])
      v = (rot.reshape(3, 3) @ v.T).T + m.geom_pos[g]
      if best is None:
        best = (v.min(0), v.max(0))
      break
  assert best is not None, "没找到 ankle mesh"
  return best


def build() -> None:
  xml = SRC_XML.read_text()

  # 1) 去掉 scene include 与 <size>（mjlab 自己控制接触槽）。
  xml = re.sub(r'\s*<include file="scene\.xml"/>', "", xml)
  xml = re.sub(r"\s*<size [^>]*/>", "", xml)

  # 1b) 删掉 <actuator> 块。
  #     mjlab 的 Entity._add_actuators() 只会往 spec 里【追加】actuator，
  #     从不清理 XML 里已有的（见 entity.py:233 与 builtin_actuator.py:450）。
  #     留着源模型的 10 个 <motor> 会与 mjlab 按 articulation 生成的 10 个
  #     <position> 并存 -> nu=20，动作维度翻倍、策略学不动。
  #     mjlab 的资产约定是 XML 里【完全不写 actuator】（14dof 模型即如此）。
  xml, n_act = re.subn(r"\s*<actuator>.*?</actuator>", "", xml, flags=re.S)
  print(f"移除源模型 actuator 块: {n_act} 个（改由 mjlab 按 articulation 生成）")

  # 2) meshdir 指向本目录下的 meshes/。
  xml = xml.replace('meshdir="../meshes/"', 'meshdir="meshes/"')

  # 3) base_link 加 freejoint + imu site。
  mo = re.search(r'(<body name="base_link"[^>]*>)', xml)
  assert mo, "没找到 base_link"
  injected = (
    mo.group(1)
    + '\n            <freejoint name="floating_base" />'
    + '\n            <site name="imu" pos="0 0 0" size="0.01" />'
  )
  xml = xml[: mo.start(1)] + injected + xml[mo.end(1) :]

  # 4) geom 命名。源模型每个 link 两个 geom：第一个是视觉（contype=0），
  #    第二个参与碰撞。转成 mjlab 约定：视觉统一 group=2，碰撞 group=3。
  def name_visual(m: re.Match[str]) -> str:
    mesh = m.group("mesh")
    return (
      f'<geom name="{mesh}_visual" type="mesh" contype="0" '
      f'conaffinity="0" group="2" rgba="{m.group("rgba")}" mesh="{mesh}" />'
    )

  def name_collision(m: re.Match[str]) -> str:
    mesh = m.group("mesh")
    return (
      f'<geom name="{mesh}_collision" type="mesh" group="3" '
      f'rgba="{m.group("rgba")}" mesh="{mesh}" />'
    )

  xml, n_vis = re.subn(
    r'<geom type="mesh" contype="0" conaffinity="0" group="1" '
    r'rgba="(?P<rgba>[^"]+)" mesh="(?P<mesh>[^"]+)" />',
    name_visual,
    xml,
  )
  xml, n_col = re.subn(
    r'<geom type="mesh" rgba="(?P<rgba>[^"]+)" mesh="(?P<mesh>[^"]+)" />',
    name_collision,
    xml,
  )

  # 5) 给两个 ankle body 加脚底胶囊 + 足部 site。
  #    ankle body 里的 mesh geom 在平地训练中不参与碰撞，降级为纯视觉。
  for side in ("right", "left"):
    body_re = re.compile(
      r'(<body name="%s_ankle_link"[^>]*>)(.*?)(\n\s*</body>)' % side,
      re.S,
    )
    mb = body_re.search(xml)
    assert mb, f"没找到 {side}_ankle_link"
    inner = mb.group(2)
    inner = inner.replace(
      f'name="{side}_ankle_link_collision" type="mesh" group="3"',
      f'name="{side}_ankle_link_collision" type="mesh" contype="0" '
      f'conaffinity="0" group="2"',
    )
    foot = [
      f'\n            <site name="{side}_foot" pos="0 0 {FOOT_SOLE_Z}" '
      f'size="0.01" rgba="1 0 0 1" />'
    ]
    for i, y in enumerate(FOOT_Y_OFFSETS, start=1):
      foot.append(
        f'\n            <geom name="{side}_foot{i}_collision" type="capsule" '
        f'size="{FOOT_RADIUS}" fromto="{-FOOT_X_HALF} {y} {FOOT_AXIS_Z} '
        f'{FOOT_X_HALF} {y} {FOOT_AXIS_Z}" group="3" '
        f'rgba="0.2 0.6 0.2 0.3" contype="1" conaffinity="1" '
        f'condim="3" priority="1" friction="0.6" />'
      )
    xml = xml[: mb.start(2)] + inner + "".join(foot) + xml[mb.end(2) :]

  print(f"命名 geom: 视觉 {n_vis} 个, 碰撞 {n_col} 个")

  # 5b) 补 <sensor> 块。
  #     imu site 用单位四元数（不写 quat）：这样 gyro/velocimeter 读到的就是
  #     base_link 系下的量，与 Entity.data.root_link_ang_vel_b / lin_vel_b 一致。
  #     （14dof 资产的 imu site 带一个 107° 的 quat，那是给已废弃的机身朝向
  #     补偿用的；v03 的 base_link 本身就是 x 前 / z 上，不需要补偿。）
  sensors = """
    <sensor>
        <framequat name="imu_quat" objtype="site" objname="imu" />
        <gyro name="imu_ang_vel" site="imu" />
        <velocimeter name="imu_lin_vel" site="imu" />
        <accelerometer name="imu_acc" site="imu" />
        <framepos name="frame_pos" objtype="site" objname="imu" />
        <framelinvel name="frame_vel" objtype="site" objname="imu" />
        <!-- 整机绕基座质心的角动量，angular_momentum 奖励用 -->
        <subtreeangmom name="root_angmom" body="base_link" />
    </sensor>
"""
  assert "</mujoco>" in xml, "没找到 </mujoco>"
  xml = xml.replace("</mujoco>", sensors + "</mujoco>")
  print("已补 <sensor> 块: imu_quat/imu_ang_vel/imu_lin_vel/imu_acc/root_angmom")

  # 6) 写出（meshes 一并复制）。
  DST_DIR.mkdir(parents=True, exist_ok=True)
  DST_MESHES.mkdir(parents=True, exist_ok=True)
  n = 0
  for f in SRC_MESHES.iterdir():
    if f.is_file():
      shutil.copy2(f, DST_MESHES / f.name)
      n += 1
  DST_XML.write_text(xml)
  print(f"复制 meshes: {n} 个文件 -> {DST_MESHES}")


def verify() -> None:
  m = mujoco.MjModel.from_xml_path(str(DST_XML))
  d = mujoco.MjData(m)
  mujoco.mj_resetData(m, d)
  mujoco.mj_forward(m, d)

  print()
  print("=== 校验 ===")
  print(f"nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} nbody={m.nbody} ngeom={m.ngeom}")

  free = [
    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(m.njnt)
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  ]
  print(f"freejoint: {free}")

  sites = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)]
  print(f"sites: {sites}")

  acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
  print(f"actuators: {acts}")

  # 站立高度：qpos=0 时脚底最低点。
  zs = []
  for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "ankle" not in name:
      continue
    did = m.geom_dataid[g]
    va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
    v = m.mesh_vert[va : va + vn].reshape(-1, 3)
    R = d.geom_xmat[g].reshape(3, 3)
    zs.append(((R @ v.T).T + d.geom_xpos[g])[:, 2].min())
  print(f"qpos=0 时足底 mesh 世界 z = {min(zs):+.5f} m")

  foot_geoms = [
    g
    for g in range(m.ngeom)
    if "_foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
  ]
  print(f"脚底胶囊 geom: {len(foot_geoms)} 个")
  print(
    f"脚底胶囊最低 z = "
    f"{min(d.geom_xpos[g][2] for g in foot_geoms) - FOOT_RADIUS:+.5f} m"
  )
  print(f"（应等于 mesh 鞋底 z，两者相差 = 建模公差）")

  # 站立高度：base_link 要抬高多少，鞋底才正好落在 z=0。
  print(f"建议 base 站立高度 = {abs(min(zs)):.5f} m")


if __name__ == "__main__":
  lo, hi = ankle_aabb()
  print(f"ankle AABB(body系) lo={np.round(lo, 4)} hi={np.round(hi, 4)}")
  build()
  print(f"已写出: {DST_XML}")
  verify()
