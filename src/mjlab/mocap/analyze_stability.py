"""从 27dof 训练体 XML 解析质量/质心/惯量，输出整机 CoM 与惯量张量。

用途：为 Pikachu 结构设计提供静态/动态稳定性诊断依据。解析每个 ``<inertial>``
的 ``mass / pos / quat / diaginertia``，经平行轴定理把各 link 的质心、惯量
合成到整机，给出：

  - 各 link 质量、质心（世界系，站立姿态）、对整机 CoM 高度的贡献；
  - 整机总质量、总 CoM（相对脚底地面的高度）；
  - 绕整机 CoM 的惯量张量及其主惯量（特征值/主轴）——判断绕各轴摆动
    的灵活度与结构对称性；
  - 静态稳定裕度：站立时 CoM 地面投影到脚底支撑多边形边界的最近距离。

用法（mjlab 目录下）：

  uv run python -m mjlab.mocap.analyze_stability [--xml PATH]

默认读取 ``pikachu_27dof_train.xml``。纯只读分析，不改任何文件。
"""

from __future__ import annotations

import argparse
import itertools
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFAULT_XML = (
  Path(__file__).resolve().parents[3]
  / "src/mjlab/asset_zoo/robots/pikachu_xml/mjcf/pikachu_sample_links_27dof.xml"
  # / "src/mjlab/asset_zoo/robots/pikachu/xmls/mjcf_14dof_lite/Pikachu_V025_flat_14dof_capsule.xml"
)


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
  """四元数 (w,x,y,z) → 旋转矩阵 R（把向量从被旋转系转到目标系）。"""
  w, x, y, z = q
  R = np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )
  return R


def _attr(elem: ET.Element, key: str, default: str = "0") -> str:
  """读取 XML 属性，None 兜底为默认字符串（ty 需要 str，不能 None）。"""
  return elem.get(key, default) or default


def _parse_inertial(
  inert: ET.Element,
) -> tuple[float, np.ndarray, np.ndarray]:
  """解析一个 <inertial>：mass、质心 (body 系)、绕质心惯性张量 (body 系)。"""
  mass = float(_attr(inert, "mass"))
  p = np.array([float(x) for x in _attr(inert, "pos", "0 0 0").split()])
  diag = np.array([float(x) for x in _attr(inert, "diaginertia").split()])
  q = np.array([1.0, 0.0, 0.0, 0.0])
  if "quat" in inert.attrib:
    q = np.array([float(x) for x in inert.attrib["quat"].split()])
    q = q / np.linalg.norm(q)
  # quat 把 body 系向量转到惯性主轴系；主轴系里惯性 = diag。
  R = _rot_from_quat(q)
  I_about_p_body = R.T @ np.diag(diag) @ R  # 绕质心、body 系坐标轴
  return mass, p, I_about_p_body


def _walk(
  body: ET.Element,
  R_body: np.ndarray,
  p_body: np.ndarray,
  out: list,
) -> None:
  """递归累积每个带 <inertial> 的 link 到 out。

  out 元素: (name, mass, com_world, I_world_about_com)。
  """
  name = body.get("name", "?")
  p_loc = np.array([float(x) for x in _attr(body, "pos").split()])
  p = p_body + R_body @ p_loc
  R: np.ndarray = R_body
  if "quat" in body.attrib:
    q = np.array([float(x) for x in _attr(body, "quat").split()])
    R = R @ _rot_from_quat(q / np.linalg.norm(q))

  for inert in body.findall("inertial"):
    m, p_com_loc, I_about_com = _parse_inertial(inert)
    p_com = p + R @ p_com_loc
    # I_about_com 已是绕 body 质心、body 系坐标轴；只旋转到世界系。
    # 平行轴统一在 analyze() 里按 d = c - com 做，这里不要再平移到 body 原点
    # （旧实现把两处参考系搞混，导致主惯量偏高 ~0.064）。
    I_world_about_com = R @ I_about_com @ R.T
    out.append((name, m, p_com, I_world_about_com))

  for child in body.findall("body"):
    _walk(child, R, p, out)


def _foot_geom_bottoms(world: ET.Element) -> tuple[float, float]:
  """扫足部的 ankle 接触盒，返回 (接触面最低 z, 侧向外缘 max|y|)（世界系）。

  沿 body 链递归把 geom 的局部 pos 转到世界，得到足盒底面与外侧 y 极值。
  用两条腿的足盒近似静态支撑多边形。
  """
  bottoms = []  # 每脚： (底面 z, |y| 外缘)
  hw = 0.0

  def walk(body: ET.Element, R: np.ndarray, o: np.ndarray) -> None:
    nonlocal hw
    o = o + R @ np.array([float(x) for x in _attr(body, "pos").split()])
    if "quat" in body.attrib:
      q = np.array([float(x) for x in _attr(body, "quat").split()])
      R = R @ _rot_from_quat(q / np.linalg.norm(q))
    if "ankle" in body.get("name", ""):
      for geom in body.findall("geom"):
        size = np.array([float(x) for x in _attr(geom, "size").split()])
        pos = o + R @ np.array([float(x) for x in _attr(geom, "pos").split()])
        bottoms.append(pos[2] - size[2])
        hw = max(hw, abs(pos[1]) + size[1])
    for child in body.findall("body"):
      walk(child, R, o)

  for b in world.findall("body"):
    walk(b, np.eye(3), np.zeros(3))
  return (min(bottoms), hw) if bottoms else (0.0, 0.0)


GR = 9.81  # m/s²，重力加速度。


def _vec(parts: str, default: str = "1 0 0 0") -> np.ndarray:
  """解析空白分隔浮点数串为数组。"""
  return np.array([float(x) for x in (parts or default).split()])


def _rot_axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
  """单位轴 axis 绕 ang 弧度的轴角旋转矩阵(Rodrigues 公式)。"""
  K = np.array(
    [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
  )
  return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _build_tree(
  world: ET.Element,
) -> tuple[list[dict], dict[str, int]]:
  """把 worldbody 下的 body 树解析成节点表 + name→idx 索引。

  每节点:parent_name、pos_local、quat_local(wxyz)、自身 inertials[(mass, com_local)]、
  joints[{name,type,pos_local,axis_local,range}]。顶层多 body 都是根。
  """
  nodes: list[dict] = []
  name_to_idx: dict[str, int] = {}

  def add(body: ET.Element, parent_name: str | None) -> None:
    idx = len(nodes)
    name = body.get("name", "?")
    name_to_idx[name] = idx
    inertials: list[tuple[float, np.ndarray]] = []
    for inert in body.findall("inertial"):
      m, p_loc, _ = _parse_inertial(inert)
      inertials.append((m, p_loc))
    joints: list[dict] = []
    for jn in body.findall("joint"):
      rng = None
      spec = jn.get("range")
      if jn.get("limited") == "true" and spec:
        lo, hi = map(float, spec.split())
        rng = (lo, hi)
      joints.append(
        {
          "name": jn.get("name", "?"),
          "type": jn.get("type", "hinge"),
          "pos_local": _vec(_attr(jn, "pos"), "0 0 0"),
          "axis_local": _vec(_attr(jn, "axis"), "0 0 0"),
          "range": rng,
        }
      )
    nodes.append(
      {
        "name": name,
        "parent_name": parent_name,
        "pos_local": _vec(body.get("pos", "0 0 0")),
        "quat_local": _vec(body.get("quat", "1 0 0 0")),
        "inertials": inertials,
        "joints": joints,
      }
    )
    for child in body.findall("body"):
      add(child, name)

  for b in world.findall("body"):
    add(b, None)
  return nodes, name_to_idx


def _fk(
  nodes: list[dict], angles: dict[str, float]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
  """给定各 body 关节角(未给默 0)，返回世界位姿 (o, R) 与各 body 自身质心世界位 com。

  ``joint.axis`` 在 body 本体系,转动使 ``R_b = R_p @ R_zero @ Rot(axis_local, θ)``。
  """
  o: dict[str, np.ndarray] = {}
  R: dict[str, np.ndarray] = {}
  com: dict[str, np.ndarray] = {}
  for node in nodes:
    pname = node.get("parent_name")
    if pname is None:
      Rp, op = np.eye(3), np.zeros(3)
    else:
      Rp, op = R[pname], o[pname]
    q0 = node["quat_local"] / np.linalg.norm(node["quat_local"])
    R_b = Rp @ _rot_from_quat(q0)
    for j in node["joints"]:
      an = np.linalg.norm(j["axis_local"])
      if j["type"] == "hinge" and an > 1e-9:
        R_b = R_b @ _rot_axis_angle(j["axis_local"] / an, angles.get(node["name"], 0.0))
    o_b = op + Rp @ node["pos_local"]
    m_b = sum(m for m, _ in node["inertials"])
    if m_b > 0:
      com_b = sum(m * (o_b + R_b @ pl) for m, pl in node["inertials"]) / m_b
    else:
      com_b = o_b.copy()
    o[node["name"]] = o_b
    R[node["name"]] = R_b
    com[node["name"]] = com_b
  return o, R, com


def _descendants(name: str, children: dict[str, list[str]]) -> list[str]:
  """返回树中严格位于 ``name`` 下游的所有 body 名。"""
  res: list[str] = []
  for c in children.get(name, []):
    res.append(c)
    res.extend(_descendants(c, children))
  return res


def _sample_angles(lo: float, hi: float) -> list[float]:
  """关节候选角:0、两端、中点、±90°(落入范围者)。"""
  cand = {0.0, lo, hi, (lo + hi) / 2.0, -math.pi / 2.0, math.pi / 2.0}
  return sorted(v for v in cand if lo - 1e-6 <= v <= hi + 1e-6)


def joint_load_torques(
  xml_path: Path,
) -> list[dict]:
  """逐关节的**最坏可达配置**静态(重力)负载力矩,计入所带全部子关节。

  关节把其所在 body 的整棵子树挂在父体上。对每个 hinge 关节:
   - 子树总质量 ``M`` 恒定 = 该 body + 所有后代 body 的 <inertial> 质量之和。
   - ``tau_max`` = M g · max|(c−o)×a|,扫**后代**关节的可达姿态找出把子树质心
     推离关节轴线最远的垂距(自身转角不改变该垂距,故只扫后代)。这等价于把重力
     摆到与该轴最正交的极端工况 —— 即第4组你点名的「arm_roll 90° / hip_pitch
     90° / ear_pitch-roll 90°」等配置下的最大静态重力矩,选型/校核扭矩上限的安全界。
   - ``tau_pose`` = 名义姿态(全关节 0)下的瞬时重力力矩 M g |a·(r×ẑ)|。
  """
  tree = ET.parse(xml_path)
  world = tree.getroot().findall(".//worldbody")[0]
  nodes, _ = _build_tree(world)

  children: dict[str, list[str]] = {n["name"]: [] for n in nodes}
  for n in nodes:
    if n["parent_name"] is not None:
      children[n["parent_name"]].append(n["name"])

  zhat = np.array([0.0, 0.0, 1.0])
  out: list[dict] = []
  rng = np.random.default_rng(0)
  for node in nodes:
    for j in node["joints"]:
      if j["type"] != "hinge":
        continue
      axis_l = j["axis_local"]
      an = np.linalg.norm(axis_l)
      if an < 1e-9:
        continue
      axis_l = axis_l / an
      name = node["name"]
      o_j = j["pos_local"]  # 关节原点在 body 本体系内。

      # 后代 hinge 关节(严格下游)—— 它们的转角决定质心到 axis 线的垂距。
      desc = []
      for dn in _descendants(name, children):
        dnode = next(n for n in nodes if n["name"] == dn)
        for dj in dnode["joints"]:
          if (
            dj["type"] == "hinge"
            and dj["range"]
            and np.linalg.norm(dj["axis_local"]) > 1e-9
          ):
            desc.append((dn, dj))
      if desc:
        per_joint = [_sample_angles(*dj["range"]) for _, dj in desc]
        ncomb = int(np.prod([len(p) for p in per_joint]))
        if ncomb <= 5000:
          combos = [
            dict(zip((d[0] for d in desc), vals, strict=False))
            for vals in itertools.product(*per_joint)
          ]
        else:
          combos = []
          for _ in range(4000):
            combos.append(
              {dn: rng.uniform(dj["range"][0], dj["range"][1]) for dn, dj in desc}
            )
        # 额外保证名义与各极限姿态被评估。
        combos.append({})
        for dn, dj in desc:
          for v in (dj["range"][0], dj["range"][1]):
            c = {d[0]: 0.0 for d in desc}
            c[dn] = v
            combos.append(c)
      else:
        combos = [{}]

      best = {"a": np.zeros(3), "o": o_j, "lever": 0.0}
      for cfg in combos:
        o, R, _ = _fk(nodes, cfg)
        o_js = o[name] + R[name] @ o_j
        a_w = R[name] @ axis_l
        subt = _subtree_com(name, nodes, children, o, R)
        r = subt - o_js
        lever = float(np.linalg.norm(np.cross(r, a_w)))
        if lever > best["lever"]:
          best = {"a": a_w, "o": o_js, "lever": lever}

      M = _subtree_mass(name, nodes, children)
      tau_max = GR * M * best["lever"]
      o0, R0, _ = _fk(nodes, {})
      o_js0 = o0[name] + R0[name] @ o_j
      a0 = R0[name] @ axis_l
      r0 = _subtree_com(name, nodes, children, o0, R0) - o_js0
      tau_pose = GR * M * abs(float(np.dot(a0, np.cross(r0, zhat))))
      out.append(
        {
          "name": j["name"],
          "type": j["type"],
          "sub_mass": M,
          "lever": best["lever"],
          "tau_pose": tau_pose,
          "tau_max": tau_max,
        }
      )
  out.sort(key=lambda x: -x["tau_max"])
  return out


def _subtree_com(
  name: str,
  nodes: list[dict],
  children: dict[str, list[str]],
  o: dict[str, np.ndarray],
  R: dict[str, np.ndarray],
) -> np.ndarray:
  """body ``name`` 整棵子树(含自身)的世界系质心。"""
  idx = {n["name"]: n for n in nodes}
  names = [name] + _descendants(name, children)
  M = 0.0
  c = np.zeros(3)
  for nm in names:
    for m, pl in idx[nm]["inertials"]:
      M += m
      c = c + m * (o[nm] + R[nm] @ pl)
  return c / M


def _subtree_mass(
  name: str, nodes: list[dict], children: dict[str, list[str]]
) -> float:
  """body ``name`` 整棵子树的惯性总质量。"""
  idx = {n["name"]: n for n in nodes}
  names = [name] + _descendants(name, children)
  return sum(m for nm in names for m, _ in idx[nm]["inertials"])


def analyze(xml_path: Path) -> dict:
  """解析 XML，返回质量/CoM/惯量诊断数据。"""
  tree = ET.parse(xml_path)
  bodies = tree.getroot().findall(".//worldbody")
  assert len(bodies) == 1
  world = bodies[0]

  links: list[tuple[str, float, np.ndarray, np.ndarray]] = []
  for b in world.findall("body"):
    _walk(b, np.eye(3), np.zeros(3), links)

  total_mass = sum(m for _, m, _, _ in links)
  com = sum(m * c for _, m, c, _ in links) / total_mass
  # 绕整机 CoM 的惯量：各 link 自己绕自身质心的惯量 + 平行轴到整机 CoM。
  I_total = np.zeros((3, 3))
  for _, m, c, I_about_com in links:
    d = c - com
    I_total += I_about_com + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
  eig, eigvec = np.linalg.eigh(I_total)
  contact_z, lat_hw = _foot_geom_bottoms(world)
  return {
    "links": links,
    "total_mass": total_mass,
    "com": com,
    "I_total": I_total,
    "principal": sorted(
      zip(eig, eigvec.T, strict=True), key=lambda e: e[0], reverse=True
    ),
    "contact_z": contact_z,
    "lateral_half_width": lat_hw,
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
  args = ap.parse_args()

  data = analyze(args.xml)

  print(f"输入模型: {args.xml}\n")
  print(f"整机总质量: {data['total_mass']:.3f} kg")
  c = data["com"]
  contact_z = data["contact_z"]
  h = c[2] - contact_z  # 相对脚底接触面的高度
  print(
    f"脚底接触面 z={contact_z:+.3f} m，足盒侧向半宽(双足外缘)={data['lateral_half_width']:.3f} m"
  )
  print(
    f"整机质心 CoM (默认姿态, 世界系): x={c[0]:+.3f} m, y={c[1]:+.3f} m, "
    f"z={c[2]:+.3f} m  ← 脚底以上的高度={h:.3f} m"
  )

  # 静态稳定裕度：CoM 地面投影到支撑外缘的距离（近似正视面矩形支撑）。
  lat_margin = data["lateral_half_width"] - abs(c[1])
  stat_angle = np.rad2deg(np.arctan2(lat_margin, h))
  if lat_margin > 0:
    print(
      f"静态侧向裕度: 侧向外缘-投影={lat_margin:.3f} m "
      f"(CoM 高 {h:.3f} m vs 半宽 {data['lateral_half_width']:.3f} m, "
      f"高/宽比={h / data['lateral_half_width']:.2f}) "
      f"→ 静态翻身角≈{stat_angle:.1f}°"
    )
  else:
    print("静态侧向裕度: < 0，CoM 投影已超出支撑外缘，默认姿态无法静态站立！")
  print("各 link 质量 / 质心高度 / 质量占比:")
  total = data["total_mass"]
  for name, m, com_l, _ in sorted(data["links"], key=lambda x: -x[1]):
    print(
      f"  {name:<30} m={m:8.4f} kg  ({m / total * 100:5.1f}%)  com_z={com_l[2]:+.3f} m"
    )

  print("\n各关节静态(重力)负载力矩 — τ_max=子树绕轴极端最大值, τ_pose=名义姿态瞬时值:")
  print(
    f"  {'关节':<24}{'子树质量(kg)':>12}{'τ_max(Nm)':>12}{'τ_pose(Nm)':>12}{'力臂(mm)':>10}"
  )
  try:
    torques = joint_load_torques(args.xml)
  except Exception as e:  # noqa: BLE001
    print(f"  (关节负载计算失败: {e})")
  else:
    for t in torques:
      print(
        f"  {t['name']:<24}{t['sub_mass']:>12.4f}{t['tau_max']:>12.4f}"
        f"{t['tau_pose']:>12.4f}{t['lever'] * 1000:>10.1f}"
      )

  print("\n绕整机 CoM 的惯量张量 (kg·m²):")
  for row in data["I_total"]:
    print("  [" + " ".join(f"{v:9.5f}" for v in row) + "]")
  print("主惯量 (特征值按大到小) + 主轴方向:")
  for val, vec in data["principal"]:
    print(f"  I={val:8.5f}  axis=({vec[0]:+.3f},{vec[1]:+.3f},{vec[2]:+.3f})")


if __name__ == "__main__":
  main()
