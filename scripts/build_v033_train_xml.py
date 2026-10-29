"""把 pikachu_v033 的展示用 MJCF 转成 mjlab 训练用 MJCF。

与 ``build_v031_train_xml.py`` 同构（源模型都是 SolidWorks -> easymjcf 的
「展示/调试」模型），v033 在 v031 的腿链（并联四杆膝、只有 ``knee_up`` 驱动）
基础上**多了两条 4-dof 手臂**，v031 踩过的坑这里一个不少：

  1. **源模型没有 ``<default>`` 块** —— joint 的 ``damping`` / ``frictionloss`` /
     ``solreflimit`` 全是 MuJoCo 默认值（前两个 0，solreflimit 用默认值）。
     必须补齐为 ``damping=0.3 frictionloss=0.01 solreflimit="0.004 1"``：
       - 没有 damping/frictionloss，四杆膝的被动杆在 PD 之外毫无耗散；
       - 没有 solreflimit，关节限位是「软」的，小腿杆件惯量只有 1e-5 量级，
         满扭矩能直接顶穿（源模型注释里自己警告过这点）。

  2. **``knee_up_joint`` 没有限位**（URDF 里 lower=upper=0，等于锁死，按
     v031 的做法忽略它）。这里按 v031 的约定补成 ``[-0.65, 1.05]``，与
     ``knee_joint`` 同范围；``knee_mid_joint`` 保持无限位（纯被动杆，行程由
     四杆闭合约束决定）。
     ⚠️ v033 实测 ``knee_up`` 与 ``knee`` 近似 1:1（knee_up ≈ knee + 1.4°），
     **正方向是屈膝**：knee_up=0 -> 60° 时鞋底相对 base 从 -0.2594 抬到
     -0.2100 m（站地时 base 下沉约 4.9 cm）。

  3. **左右轴约定与 v031 又不同**（实测 ``m.jnt_axis``）：

       关节         右轴              左轴
       hip_pitch    [0,-1, 0]         [0, 1, 0]      反号
       hip_roll     [1, 0, 0]         [1, 0, 0]      同号
       hip_yaw      [-0.36315, 0, 0.931731]  两侧相同（斜轴，非竖直）
       knee_up      [0,-1, 0]         [0,-1, 0]      同号
       knee_mid     [0,-1, 0]         [0,-1, 0]      同号
       knee         [0,-1, 0]         [0, 1, 0]      反号
       ankle        [0, 1, 0]         [0,-1, 0]      反号
       arm_pitch    [0, 1, 0]         [0,-1, 0]      反号
       arm_roll     [1, 0, 0]         [1, 0, 0]      同号
       arm_yaw      [0,-1, 0]         [0, 1, 0]      反号
       elbow        [1, 0, 0]         [1, 0, 0]      同号

     训练模型直接沿用源模型的轴，不做任何镜像改写。

  4. 训练模型与源模型**同目录**（``xmls/pikachu_v033/mjcf/``），``meshdir`` 仍是
     ``../meshes/``，所以不必复制一份 10 MB 的 STL。

  5. **保留两个 ``<keyframe>``**（用户要求，供之后手工定义四足姿态）：
       - ``stand``：双足站立，qpos 全 0，base 抬到鞋底刚好贴地（实测 0.25941 m）；
       - ``quad`` ：四足姿态**占位**，当前与 ``stand`` 相同，qpos 留给用户改。
     注意：mjlab 训练时真正使用的初始状态来自
     ``pikachu_v033_constants.py`` 的 ``BIPED_KEYFRAME`` / ``QUAD_KEYFRAME``，
     XML 里的 keyframe 只用于 MuJoCo viewer 预览；改四足姿态时两边都要改。

其余转换与 v031 相同：加 freejoint / 命名 geom / 加足部胶囊与 site / 删 scene
include / 删 <size> / 删 <actuator> 块 / 补 <sensor> 块。

用法：
    uv run python scripts/build_v033_train_xml.py
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

SRC_XML = (
  Path(__file__).resolve().parent.parent
  / "src"
  / "mjlab"
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "pikachu_v033"
  / "mjcf"
  / "pikachu_v033.xml"
)
DST_XML = SRC_XML.with_name("pikachu_v033_train.xml")

# 与 v03 / v031 一致：源模型缺 <default>，训练模型补上。
DEFAULT_BLOCK = """
    <default>
        <joint damping="0.3" frictionloss="0.01" solreflimit="0.004 1" />
    </default>
"""

# knee_up 限位：沿用 v031 的约定（与 knee 同范围）。
# knee_mid 保持无限位（纯被动杆，行程由四杆闭合约束决定）。
KNEE_UP_RANGE = "-0.65 1.05"

# 脚底胶囊参数：由 v033 ankle mesh 在 body 系下的 AABB 实测得到：
# x ±0.0465（脚长 9.3 cm）、y -0.0354/+0.0356（脚宽 7.1 cm）、鞋底 z=-0.034。
# 与 v031 的脚底胶囊完全一致，直接沿用。
FOOT_SOLE_Z = -0.034
FOOT_RADIUS = 0.012
# 胶囊的 fromto 是【轴线】，碰撞面在轴线外 FOOT_RADIUS 处。
# 要让胶囊底面正好落在 mesh 鞋底平面上，轴线必须抬高一个半径。
FOOT_AXIS_Z = FOOT_SOLE_Z + FOOT_RADIUS
FOOT_X_HALF = 0.040
FOOT_Y_OFFSETS = (-0.028, -0.014, 0.0, 0.014, 0.028)


def build() -> None:
  xml = SRC_XML.read_text()

  # 1) 去掉 scene include 与 <size>（mjlab 自己控制接触槽）。
  xml = re.sub(r'\s*<include file="scene\.xml"/>', "", xml)
  xml = re.sub(r"\s*<size [^>]*/>", "", xml)

  # 1b) 删掉 <actuator> 块。
  #     mjlab 的 Entity._add_actuators() 只会往 spec 里【追加】actuator，
  #     从不清理 XML 里已有的。留着源模型的 18 个 <motor> 会与 mjlab 按
  #     articulation 生成的 18 个 <position> 并存 -> nu=36，动作维度翻倍。
  xml, n_act = re.subn(r"\s*<actuator>.*?</actuator>", "", xml, flags=re.S)
  print(f"移除源模型 actuator 块: {n_act} 个（改由 mjlab 按 articulation 生成）")

  # 1c) 补 <default>：源模型没有，缺了 damping/frictionloss 标定时四杆被动杆
  #     无耗散，缺了 solreflimit 关节限位顶不住。
  assert "<default>" not in xml, "源模型已自带 <default>，请人工核对"
  mo = re.search(r"(<compiler[^>]*/>)", xml)
  assert mo, "没找到 <compiler>"
  xml = xml[: mo.end(1)] + DEFAULT_BLOCK + xml[mo.end(1) :]
  print("补 <default>: joint damping=0.3 frictionloss=0.01 solreflimit=0.004 1")

  # 1d) knee_up 限位（源模型两个 knee_up 都没有 limited/range）。
  xml, n_ku = re.subn(
    r'<joint name="((?:right|left)_knee_up_joint)" pos="0 0 0" axis="0 -1 0" />',
    lambda m: (
      f'<joint name="{m.group(1)}" pos="0 0 0" axis="0 -1 0" '
      f'limited="true" range="{KNEE_UP_RANGE}" />'
    ),
    xml,
  )
  assert n_ku == 2, f"knee_up 限位补丁命中 {n_ku} 处，应为 2"
  print(f"补 knee_up 限位: range={KNEE_UP_RANGE}（knee_mid 保持无限位）")

  # 2) base_link 加 freejoint + imu site。
  mo = re.search(r'(<body name="base_link"[^>]*>)', xml)
  assert mo, "没找到 base_link"
  injected = (
    mo.group(1)
    + '\n            <freejoint name="floating_base" />'
    + '\n            <site name="imu" pos="0 0 0" size="0.01" />'
  )
  xml = xml[: mo.start(1)] + injected + xml[mo.end(1) :]

  # 3) geom 命名。源模型每个 link 两个 geom：第一个是视觉（contype=0），
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
  assert n_vis == n_col == 23, f"期望 23 个 link 各一对 geom，实得 {n_vis}/{n_col}"

  # 4) 给两个 ankle body 加脚底胶囊 + 足部 site。
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

  # 4b) 补 <sensor> 块。imu site 用单位四元数：gyro/velocimeter 读到的就是
  #     base_link 系下的量，与 Entity.data.root_link_ang_vel_b / lin_vel_b 一致。
  sensors = """
    <sensor>
        <framequat name="imu_quat" objtype="site" objname="imu" />
        <gyro name="imu_ang_vel" site="imu" />
        <velocimeter name="imu_lin_vel" site="imu" />
        <accelerometer name="imu_acc" site="imu" />
        <framepos name="frame_pos" objtype="site" objname="imu" />
        <!-- frame* 系列只认 objtype/objname，不像 gyro/velocimeter 那样有 site。 -->
        <framelinvel name="frame_vel" objtype="site" objname="imu" />
        <!-- 整机绕基座质心的角动量，angular_momentum 奖励用 -->
        <subtreeangmom name="root_angmom" body="base_link" />
    </sensor>
"""
  assert "</mujoco>" in xml, "没找到 </mujoco>"
  xml = xml.replace("</mujoco>", sensors + "</mujoco>")
  print("已补 <sensor> 块: imu_quat/imu_ang_vel/imu_lin_vel/imu_acc/root_angmom")

  # 先写出不含 keyframe 的版本，编译后量出 nq / 站立高度。
  DST_XML.write_text(xml)
  m = mujoco.MjModel.from_xml_path(str(DST_XML))
  d = mujoco.MjData(m)
  mujoco.mj_resetData(m, d)
  mujoco.mj_forward(m, d)

  stand_height = -_lowest_foot_z(m, d)
  print(
    f"qpos=0 时鞋底世界 z = {-stand_height:+.5f} m -> base 站立高度 = {stand_height:.5f} m"
  )

  # 5) 两个 keyframe：stand（双足站姿）+ quad（四足占位，待用户填写）。
  qpos = np.zeros(m.nq)
  qpos[0:3] = (0.0, 0.0, stand_height)
  qpos[3] = 1.0  # 四元数 wxyz，单位四元数
  stand_qpos = " ".join(f"{v:g}" for v in qpos)
  # quad 先与 stand 相同，用户之后按四足姿态改这里的 qpos。
  quad_qpos = stand_qpos

  keyframes = f"""
    <keyframe>
        <!-- 双足站立：qpos 全 0，base 抬到鞋底刚好贴地。 -->
        <key name="stand" qpos="{stand_qpos}" />
        <!-- 四足姿态占位：TODO 用户填写（base 前俯角 + 髋/膝/踝/臂角度）。
             注意 mjlab 训练读的是 pikachu_v033_constants.py 的 QUAD_KEYFRAME，
             改姿态时两边同步。 -->
        <key name="quad" qpos="{quad_qpos}" />
    </keyframe>
"""
  xml = xml.replace("</mujoco>", keyframes + "</mujoco>")
  DST_XML.write_text(xml)
  print(f"已写出: {DST_XML}")


def _lowest_foot_z(m: mujoco.MjModel, d: mujoco.MjData) -> float:
  """当前位姿下两侧 ankle mesh 的最低世界 z。"""
  zs = []
  for side in ("right", "left"):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_link")
    for g in range(m.ngeom):
      if m.geom_bodyid[g] != b:
        continue
      if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
      did = m.geom_dataid[g]
      va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
      v = m.mesh_vert[va : va + vn].reshape(-1, 3)
      R = d.geom_xmat[g].reshape(3, 3)
      zs.append(((R @ v.T).T + d.geom_xpos[g])[:, 2].min())
  assert zs, "没找到 ankle mesh"
  return float(min(zs))


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

  keys = [m.key(k).name for k in range(m.nkey)]
  print(f"keyframes: {keys}")
  for k in range(m.nkey):
    print(
      f"  {m.key(k).name}: qpos[:7]={np.round(m.key_qpos[k][:7], 5)} "
      f"nq_key={m.key_qpos.shape[1]}"
    )

  print(f"total mass = {m.body_mass.sum():.6f} kg")
  print(f"dof_damping 唯一值 = {sorted(set(np.round(m.dof_damping, 4)))}")
  print(f"dof_frictionloss 唯一值 = {sorted(set(np.round(m.dof_frictionloss, 5)))}")

  for n in (
    "right_knee_up_joint",
    "left_knee_up_joint",
    "right_knee_mid_joint",
    "right_knee_joint",
  ):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    print(f"  {n}: limited={m.jnt_limited[j]} range={m.jnt_range[j]}")

  print(f"qpos=0 时鞋底 mesh 世界 z = {_lowest_foot_z(m, d):+.5f} m")
  print(f"建议 base 站立高度 = {abs(_lowest_foot_z(m, d)):.5f} m")


if __name__ == "__main__":
  build()
  verify()
