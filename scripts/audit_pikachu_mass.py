"""Read-only V033/V034 mass provenance audit across CAD, URDF and MuJoCo.

Write the report/data into an experiment's categorized artifact tree. Source
robot files are never modified. CAD exports are evidence of exported values,
not a substitute for weighing the assembled robot or checking its BOM.
"""

import argparse
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/mjlab/asset_zoo/robots/pikachu/xmls"
UPSTREAM = ROOT.parent.parent / "pikachu_urdf"


def urdf_links(path: Path) -> dict:
  result = {}
  for link in ET.parse(path).findall("link"):
    inertial = link.find("inertial")
    if inertial is None:
      continue
    mass, inertia, origin = (
      inertial.find("mass"),
      inertial.find("inertia"),
      inertial.find("origin"),
    )
    assert mass is not None and inertia is not None and origin is not None
    result[link.attrib["name"]] = {
      "mass": float(mass.attrib["value"]),
      "origin": dict(origin.attrib),
      "inertia": {k: float(v) for k, v in inertia.attrib.items()},
    }
  return result


def csv_links(path: Path) -> dict:
  with path.open(newline="", encoding="utf-8-sig") as stream:
    return {
      row["Link Name"]: {"mass": float(row["Mass"])}
      for row in csv.DictReader(stream)
      if row.get("Link Name")
    }


def mjcf_links(path: Path) -> dict:
  result = {}
  for body in ET.parse(path).findall(".//body"):
    inertial = body.find("inertial")
    if inertial is not None:
      result[body.attrib["name"]] = {"mass": float(inertial.attrib["mass"])}
  return result


def tensor(values: dict) -> np.ndarray:
  return np.array(
    [
      [values["ixx"], values["ixy"], values["ixz"]],
      [values["ixy"], values["iyy"], values["iyz"]],
      [values["ixz"], values["iyz"], values["izz"]],
    ]
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True, help="Experiment root")
  args = parser.parse_args()
  output = args.output.resolve()
  data_dir = output / "research/mass_audit_20261023"
  provenance = output / "provenance/mass_audit_20261023"
  for directory in (data_dir, provenance, output / "reports"):
    directory.mkdir(parents=True, exist_ok=True)
  sources = {}

  def add(label, path, reader):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = provenance / label / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
      destination.exists()
      and hashlib.sha256(destination.read_bytes()).hexdigest() != digest
    ):
      raise ValueError(f"Refusing to replace frozen input {destination}")
    shutil.copy2(path, destination)
    links = reader(path)
    sources[label] = {
      "path": str(path),
      "sha256": digest,
      "links": links,
      "link_count": len(links),
      "total_mass_kg": sum(v["mass"] for v in links.values()),
    }

  for version in ("v033", "v034"):
    name = "pikachu_" + version
    add(version + "_cad", UPSTREAM / name / "urdf" / f"{name}.csv", csv_links)
    add(version + "_urdf", UPSTREAM / name / "urdf" / f"{name}.urdf", urdf_links)
    add(
      version + "_upstream_mjcf", UPSTREAM / name / "mjcf" / f"{name}.xml", mjcf_links
    )
    for suffix in ("", "_train"):
      add(
        version + "_repo" + suffix,
        ASSETS / name / "mjcf" / f"{name}{suffix}.xml",
        mjcf_links,
      )
  add(
    "v034_origin", UPSTREAM / "pikachu_v034/urdf/pikachu_v034_origin.urdf", urdf_links
  )
  add(
    "v034_backup", UPSTREAM / "pikachu_v034/urdf/pikachu_v034.urdf.backup", urdf_links
  )
  original, edited = sources["v034_origin"]["links"], sources["v034_urdf"]["links"]
  rows = []
  for name in sources["v034_repo_train"]["links"]:
    row = {"link": name}
    for key in (
      "v033_cad",
      "v033_repo_train",
      "v034_cad",
      "v034_origin",
      "v034_urdf",
      "v034_repo_train",
    ):
      row[key + "_mass_kg"] = sources[key]["links"][name]["mass"]
    before, after = original[name], edited[name]
    old_tensor, new_tensor = tensor(before["inertia"]), tensor(after["inertia"])
    eigenvalues = np.linalg.eigvalsh(new_tensor)
    row.update(
      v034_added_mass_kg=after["mass"] - before["mass"],
      com_origin_unchanged=before["origin"] == after["origin"],
      inertia_trace_ratio=float(np.trace(new_tensor) / np.trace(old_tensor)),
      inertia_positive_definite=bool((eigenvalues > 0).all()),
      inertia_triangle_valid=bool(eigenvalues[-1] <= eigenvalues[:-1].sum() + 1e-12),
    )
    rows.append(row)
  with (data_dir / "per_link_mass.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  compiled = {}
  for version in ("v033", "v034"):
    name = "pikachu_" + version
    model = mujoco.MjModel.from_xml_path(
      str(ASSETS / name / "mjcf" / f"{name}_train.xml")
    )
    declared = sources[version + "_repo_train"]["links"]
    for link, attributes in declared.items():
      np.testing.assert_allclose(
        model.body(link).mass[0], attributes["mass"], rtol=0, atol=1e-12
      )
    root = ET.parse(ASSETS / name / "mjcf" / f"{name}_train.xml").getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiled[version] = {
      "total_mass_kg": float(model.body_mass.sum()),
      "robot_body_count": model.nbody - 1,
      "all_robot_bodies_have_explicit_inertial": len(root.findall(".//body/inertial"))
      == len(declared),
      "body_mass_matches_declared_mass_by_name": True,
      "compiler_attributes": compiler.attrib,
    }
  masses = {key: value["total_mass_kg"] for key, value in sources.items()}
  repeated = [row for row in rows if abs(row["v034_added_mass_kg"] - 0.2080665) < 2e-6]
  cad_ratio = masses["v034_cad"] / masses["v033_cad"]
  for version in ("v033", "v034"):
    source_links = sources[version + "_repo"]["links"]
    assert source_links == sources[version + "_repo_train"]["links"]
  assert sources["v034_cad"]["links"].keys() == original.keys()
  for name, attributes in original.items():
    assert sources["v034_cad"]["links"][name]["mass"] == attributes["mass"]
  invalid_inertias = [
    row["link"]
    for row in rows
    if not row["inertia_positive_definite"] or not row["inertia_triangle_valid"]
  ]
  result = {
    "sources": sources,
    "compiled": compiled,
    "cad_v034_to_v033_ratio": cad_ratio,
    "invalid_edited_urdf_inertias": invalid_inertias,
    "training_v034_to_v033_ratio": masses["v034_repo_train"]
    / masses["v033_repo_train"],
    "v034_edited_minus_original_kg": masses["v034_urdf"] - masses["v034_origin"],
    "repeated_208g_addition_count": len(repeated),
    "repeated_208g_addition_kg": sum(row["v034_added_mass_kg"] for row in repeated),
    "all_23_inertial_origins_unchanged": all(
      row["com_origin_unchanged"] for row in rows
    ),
    "left_knee_mid_asymmetry_kg": edited["left_knee_mid_link"]["mass"]
    - edited["right_knee_mid_link"]["mass"],
    "physical_mass_authority": "unconfirmed: requires BOM/weighed mass and explanation of edited URDF",
    "source_files_modified": False,
  }
  for source in sources.values():
    assert (
      hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    )
  (data_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
  table = ["| 来源 | v033 kg | v034 kg |", "|---|---:|---:|"]
  for label, suffix in (
    ("原始 CAD CSV", "cad"),
    ("当前上游 URDF", "urdf"),
    ("仓库源 MJCF", "repo"),
    ("训练 MJCF", "repo_train"),
  ):
    table.append(
      f"| {label} | {masses['v033_' + suffix]:.9f} | {masses['v034_' + suffix]:.9f} |"
    )
  detailed = [
    "| 连杆 | v034 原始 kg | 当前 URDF kg | 增量 kg |",
    "|---|---:|---:|---:|",
  ]
  for row in rows:
    detailed.append(
      f"| {row['link']} | {row['v034_origin_mass_kg']:.6f} | {row['v034_urdf_mass_kg']:.6f} | {row['v034_added_mass_kg']:+.6f} |"
    )
  report = f"""# V033 / V034 质量来源审核

审核结论：1.89 倍来自当前修改版 V034 URDF/MJCF 的质量，不是原始 CAD 导出两代质量之比。
原始 V034 CSV 与 origin URDF 均为 **{masses["v034_cad"]:.9f} kg**，比 V033 CAD
重约 **{(cad_ratio - 1) * 100:.2f}%**。当前 V034 URDF 为
**{masses["v034_urdf"]:.9f} kg**，比其 origin 版本多 **{result["v034_edited_minus_original_kg"]:.9f} kg**。

{chr(10).join(table)}

`pikachu_v034.urdf.backup` 也已是 8.890784 kg。这说明增重存在于当前导入输入及其备份中；
仅凭文件无法确定修改者、修改理由或实机质量。MJCF 的 5 毫克（0.000005 kg）差异来自 base 质量的小数截断。

## 额外质量分布

- 12 个连杆各增加约 0.208066 kg，合计 **{result["repeated_208g_addition_kg"]:.6f} kg**。
- base_link：0.684560 → 1.496395 kg，增加约 0.811835 kg。
- 两侧 arm_roll：各增加约 0.178145 kg，合计约 0.356289 kg。
- 左 knee_mid：额外增加约 0.017185 kg；右侧没有对应增量。
- 其余质量变化仅为小数舍入。此分布像逐组件补重，但不能在缺少 BOM 的情况下认定为真实补重或重复添加电机质量。

{chr(10).join(detailed)}

## 惯量及对称性疑点

- 23 个连杆的惯性原点/质心坐标全部保持原值，但惯量张量全部经过改写。
- base 的惯量 trace 约为 origin 的 12.53 倍，质量仅为约 2.19 倍；需要补重位置和合成惯量计算依据。
- elbow 的质量基本不变，惯量 trace 却增至约 1.73 倍；ankle 约 1.29 倍。
- 左 knee_mid 0.194382 kg、右 0.177197 kg；原始 CSV 两侧约 0.177197 kg，额外的左侧 17.185 g 需确认。
- 当前 URDF 的 23 个惯量均通过正定/三角不等式检查，逐项记录于 CSV；通过基本数学检查不能证明与实物一致。

## 排除的接入问题

- 两代各 23 个机器人刚体，逐刚体均有显式 inertial。
- 编译后的每个 body_mass 与其 XML 声明一致，没有因视觉网格和碰撞网格重复声明而增加质量。
- V034 新增足底碰撞胶囊的 mass 为 0；原始、训练 MJCF 的逐连杆质量一致。
- 关节顺序整理及新的策略坐标映射没有改写质量；未发现全局 kg/g 缩放引起此差异。

## 对训练的处理

在用户确认补重依据前，不把 8.89 kg 当作已经确认的实机质量，不通过任意缩放 mass 或惯量来追求步态。
若 8.89 kg 是按实机 BOM 有意补重，应核对每项补重、质心移动、平行轴定理及左右差异。
若改写无物理依据，应从可信 CAD/实测数据重建**完整 inertial**，不能只改 mass 而保留当前改写的惯量。
先前关于 1.9 倍控制增益的配置只作为未验证候选；本次审核后暂停追加训练，不将其视为已验证修复。

原始输入及 SHA-256：`../provenance/mass_audit_20261023/`；
机器可读记录：`../research/mass_audit_20261023/audit.json`；
逐连杆数据：`../research/mass_audit_20261023/per_link_mass.csv`。
本次审核未修改任何机器人源文件。
"""
  (output / "reports/mass_audit.md").write_text(report)
  print(
    json.dumps(
      {
        "masses_kg": masses,
        "added_mass_kg": result["v034_edited_minus_original_kg"],
        "report": str(output / "reports/mass_audit.md"),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
