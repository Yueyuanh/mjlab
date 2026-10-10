# 01 · 模型转换：源 MJCF → 训练 MJCF

**脚本**：`scripts/build_v03_train_xml.py`
**源**：`pikachu_urdf/pikachu_v03_base_series/mjcf/pikachu_v03_base_series.xml`
**目标**：`src/mjlab/asset_zoo/robots/pikachu/xmls/mjcf_v03/pikachu_v03_base_series_train.xml`

源模型是 SolidWorks → `easymjcf` 转出的**展示/调试**模型：底座焊死在世界、
geom 无名字、没有足部 site。训练用模型需要做 6 处改造。

---

## 坑 1 · 必须删掉源模型的 `<actuator>` 块

`Entity._add_actuators()`（`entity.py:233` → `builtin_actuator.py:450`）
**只会往 spec 里追加 actuator，从不清理 XML 里已有的**。

源模型自带 10 个 `<motor>`。如果不删，mjlab 再按 articulation 生成 10 个
`<position>`，`nu = 20` —— 动作维度翻倍，策略直接学不动。

mjlab 的资产约定是 **XML 里完全不写 actuator**（14dof / 27dof 资产都如此）。

```python
xml, n_act = re.subn(r"\s*<actuator>.*?</actuator>", "", xml, flags=re.S)
```

验证：`nu == 10`。

---

## 坑 2 · 必须补 `<sensor>` 块（含 imu）

velocity 任务的观测直接读 XML 传感器：

| 观测/奖励 | 传感器 | 类型 |
|---|---|---|
| `base_lin_vel`（actor 观测） | `robot/imu_lin_vel` | `velocimeter` |
| `base_ang_vel`（actor 观测） | `robot/imu_ang_vel` | `gyro` |
| `angular_momentum`（奖励） | `robot/root_angmom` | `subtreeangmom` |

源模型没有这些（14dof 资产是靠 `include sensors.xml` 提供的）。
缺了会在建环境时直接：

```
KeyError: Scene element 'robot/imu_lin_vel' not found
```

补法：

```xml
<sensor>
    <framequat name="imu_quat" objtype="site" objname="imu" />
    <gyro name="imu_ang_vel" site="imu" />
    <velocimeter name="imu_lin_vel" site="imu" />
    <accelerometer name="imu_acc" site="imu" />
    <framepos name="frame_pos" objtype="site" objname="imu" />
    <framelinvel name="frame_vel" objtype="site" objname="imu" />
    <subtreeangmom name="root_angmom" body="base_link" />
</sensor>
```

### 坑 2b · imu site 必须是**单位四元数**

`gyro` / `velocimeter` 读的是 **site 坐标系**下的量。如果 site 的 `quat`
不是单位四元数，读数就落在被旋转过的坐标系里 —— 策略看到的「机身速度」
与真实 base 系速度不符。**这是最难察觉的一类 bug**：不报错、不 NaN，
只是训练效果莫名变差。

> 14dof 资产的 `imu` site 带一个约 107° 的 `quat`，那是给已废弃的机身朝向
> 补偿用的。v03 的 `base_link` 本身就是 x 前 / z 上，**不要照抄**。

所以 v03 的 imu site 写成 `<site name="imu" pos="0 0 0" size="0.01" />`，
不写 `quat`。

冒烟测试里有一项专门校验这个：

```
imu_ang_vel vs root_link_ang_vel_b 最大偏差: 0.000e+00 rad/s
imu_lin_vel vs root_link_lin_vel_b 最大偏差: 0.000e+00 m/s
✓ imu site 为单位四元数，观测帧正确
```

---

## 坑 3 · `CollisionCfg` 会把没匹配到的 geom 碰撞清零

`CollisionCfg.disable_other_geoms` 默认 `True`：所有**未被 `geom_names_expr`
匹配**的 geom，`contype`/`conaffinity` 都会被清零。

平地训练用 `FEET_ONLY_COLLISION`，只匹配 `^(left|right)_foot[1-5]_collision$`。
这带来一个副作用（是好事）：源模型里那些高面数 mesh 碰撞体
（`base_link.STL` 近 10 MB 三角形）不参与接触，不拖慢仿真。

**但**这要求脚底胶囊必须真的存在且命名匹配 —— 见坑 4。

---

## 坑 4 · 脚底是 5 条胶囊，底面要与 mesh 鞋底**精确同高**

v03 的脚底不是 V025 的 7 条脚趾胶囊，是 5 条（`foot1..5`），由 ankle mesh
在 body 系下的实测 AABB 生成：

```
x ±0.0465（脚长 9.3 cm）、y ±0.0355（脚宽 7.1 cm）、鞋底 z = -0.034
```

关键：**胶囊的 `fromto` 是轴线，碰撞面在轴线外一个半径处**。要让胶囊底面
正好落在 mesh 鞋底平面上，轴线必须抬高一个半径：

```python
FOOT_SOLE_Z = -0.034  # mesh 鞋底
FOOT_RADIUS = 0.012
FOOT_AXIS_Z = FOOT_SOLE_Z + FOOT_RADIUS  # = -0.022
```

写错会让机器人整体悬空/陷地一个半径（1.2 cm），而站立高度是按鞋底算的 ——
症状是「开环站姿比 keyframe 高/低 1.2 cm」，很容易误判成 PD 刚度问题。

同时 ankle body 里原来的 mesh geom 要降级为纯视觉（`contype=0
conaffinity=0`），否则会和胶囊双重接触。

---

## 坑 5 · 踝关节等效惯量只有 9.2e-5，刚性执行器要靠 `implicitfast` 才稳

`scripts/diag_v03_stability.py` 打印 `mj_fullM` 的对角元：

| 关节 | `M[dof,dof]` |
|---|---|
| `hip_pitch` | 1.80e-2 |
| `hip_roll` | 1.24e-2 |
| `hip_yaw` | 1.06e-3 |
| `knee` / `knee_up` | 1.46e-3 / 3.82e-4 |
| **`ankle`** | **9.21e-5** |

踝关节比 `hip_pitch` 小**两个数量级**。给它挂刚性 `position` 执行器
（kp=150），显式积分下 MuJoCo 第 ~21 步就报：

```
WARNING: Nan, Inf or huge value in QACC at DOF 5. The simulation is unstable. Time = 0.1040.
```

（DOF 5 = `right_ankle_joint`，t=0.104 s ≈ 21 步 × 0.005 s。）

**真正把它救活的是积分器，不是 `armature`**。同一套 kp=150 跑 2000 步
（`diag_v03_stability.py` 最后一组，可直接复现）：

| 积分器 | `armature` | `max\|qvel\|` |
|---|---|---|
| `euler` | 无 | 3.11e+00（振铃） |
| `euler` | +J3507 | 9.80e+00（更差） |
| **`implicitfast`** | 无 | **6.3e-11** |
| **`implicitfast`** | +J3507 | **3.1e-11** |

mjlab 的默认就是 `implicitfast`（`src/mjlab/sim/sim.py:103`），v03 任务没有覆盖它
—— 所以**这个坑在训练里是被默认值挡掉的**，换 `euler` 就会踩。

那 `armature` 干什么？**管的是物理保真度，不是数值稳定**：

```python
# pikachu_v03_constants.py
ROTOR_INERTIAS_J3507 = 6.69169e-6
GEARS_J3507 = (1, 7.0, 3.0)  # 7×3 = 21
ARMATURE_J3507 = reflected_inertia(6.69169e-6, 21)  # = 2.95e-3
STIFFNESS_J3507 = ARMATURE_J3507 * NATURAL_FREQ**2  # 刚度由 armature 反推
```

`2.95e-3` 是踝关节连杆惯量（9.2e-5）的 **32 倍** —— 真机上电机转子反射回来的
惯量本来就该主导这个关节。刚度 `kp` 也是从它按 `ARMATURE × ω²` 反推的，
不是拍脑袋填的数。

> ⚠️ **`armature` 不在 XML 里**。它在 `pikachu_v03_constants.py` 的
> `ElectricActuator(reflected_inertia=...)`，由 mjlab 运行时注入。
> `grep armature` 训练 XML 是**空的** —— 我第一次就是这么找错的。

> **这个脚本最初的假设是错的**，留在这里当教训：原 docstring 写「源模型本身在
> `mj_step` 下发散」，重跑后证伪 —— 源模型（base 焊死在世界、无执行器）
> **开着重力和关着重力都能稳跑 2000 步**（`max|qvel| = 2.2e-2`）。
> 唯一不稳的组合是「刚性执行器 + 显式积分」。
> **先跑最小复现，再写归因。**

---

## 其它改造

1. **`base_link` 加 `<freejoint name="floating_base"/>`** —— 否则机器人不能移动。
2. **geom 显式命名** —— `*_visual`（group 2）/ `*_collision`（group 3），
   mjlab 的 `CollisionCfg` 靠正则匹配。
3. **足部 site** `left_foot` / `right_foot`（`pos="0 0 -0.034"`）——
   `foot_clearance` / `foot_slip` 奖励和足底高度扫描传感器都依赖它。
4. **删 `<include file="scene.xml"/>`** —— mjlab 的地面由 `TerrainEntity`
   提供，保留资产自带 floor 会与地形重叠产生双重地面。
5. **删 `<size>`** —— mjlab 自己控制接触槽。
6. **`meshdir` 改 `../meshes/` → `meshes/`**，并把 meshes 一并复制过去。

---

## 验收（`build_v03_train_xml.py` 的 `verify()`）

```
nq=21 nv=20 nu=0 njnt=15 nbody=16 ngeom=40
freejoint: ['floating_base']
sites: ['imu', 'left_foot', 'right_foot']
actuators: []
脚底胶囊 geom: 10 个
qpos=0 时足底 mesh 世界 z = -0.24495 m
```

`nu=0` 是**正确**的 —— actuator 由 mjlab 在运行时按 articulation 追加，
追加后 `nu=10`（冒烟测试确认）。
