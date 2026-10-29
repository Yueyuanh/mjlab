# Pikachu 足底遥测录制数据说明

> **面向**：拿 `metrics_v03_*.csv` 做判读、画图、写报告的人。
> **覆盖**：`add_pikachu_foot_telemetry_metrics`（`src/mjlab/tasks/pikachu_foot_telemetry.py`）
> 与 `add_pikachu_telemetry_metrics`（`src/mjlab/tasks/pikachu_metrics.py`）注册的
> 全部 **122 个足底相关列**。
> **逐列规格**（定义/公式/单位/参考系/实测值/判读/坑）见 [COLUMNS.md](COLUMNS.md)。
> **展示形式**（每组该画什么图、为什么）见 [DISPLAY.md](DISPLAY.md)。

---

## 0. 先读这一条

**这批数据的 `left` / `right` 后缀不是一回事。**

其中 14 个 per-side 列（几何、姿态、CoP 共 26 个 CSV 列 + 压力 50 列）的
`/left` 实际描述的是**名叫 `right_ankle_link` 的那只脚**，与传感器列
（`foot_contact`、`foot_force_*` 等）的 `/left` **指的不是同一只脚**。
细节、证据与影响清单见 [第 3.1 节](#31-️-taxel-派生列的-leftright-与物理脚相反)。

任何跨列的左右对比、任何"左脚…右脚…"的结论，在修好之前都必须先把这 14 个
stem 的左右对调。

---

## 1. 数据从哪来

### 1.1 生成：两个注册函数

| 模块 | 注册函数 | 列数 | 内容 |
|---|---|---|---|
| `src/mjlab/tasks/pikachu_foot_telemetry.py` | `add_pikachu_foot_telemetry_metrics` | 119 | 36 个 stem：28 个 per-side stem（×2）+ `foot_pressure_pa`（×50）+ `foot_ground_friction_mu`、`base_roll_deg`、`base_roll_rate_rad_s`（各 ×1）+ `joint_torque_rate_nm_s`（×关节数，v031 为 10） |
| `src/mjlab/tasks/pikachu_metrics.py` | `add_pikachu_telemetry_metrics` | 3（足底相关） | `foot_slip_velocity_m_s`、`foot_slide_fraction`、`foot_lateral_separation_m`（各单列） |

两者都只在 **play / 录制**路径注册（训练时不注册），且注册是幂等的。
另有 `base_height_m`、`action_acc`、`actuator_spec_violation` 与 10 关节 × 8 族
（`joint_position_rad/*` 等 80 列）同源注册，但**不属于足底遥测**，本文不展开。

### 1.2 录制：`scripts/record_v03_metrics.py`

```bash
MUJOCO_GL=egl uv run python scripts/record_v03_metrics.py \
    --checkpoint <model_xxxx.pt> --out <dir> --tag <tag> --clips fwd --seconds 4
```

- **单 env**（`cfg.scene.num_envs = 1`）、固定命令、关课程、`episode_length_s = 1e9`
  （只会因摔倒终止）。
- 每档一个 CSV：`metrics_v03_<clip>_<tag>.csv`，另写一份 `metrics_v03_<tag>_index.json`。
- `--mode` 决定噪声/推力/启动 DR：`nominal`（全关）/ `dr`（默认，保留启动 DR）/ `train`。

### 1.3 帧与列的结构

- **一行 = 一个控制步**。`timestep = 0.005 s`、`decimation = 4` → **控制频率 50 Hz**，
  `step_dt = 0.02 s`，每个控制步内含 4 个 5 ms 物理子步。
  （计时器类列如 `foot_stance_time_s` 以 **0.005 s** 为量化步长，因为它们在物理子步上累加。）
- 前 5 列固定为 `sample, sim_step, time_s, env_idx, network`，其后按
  `env.metrics_manager.active_terms` 顺序排。
- **逐帧 CSV 拿到的永远是当帧真值**。`_REDUCE` 里那些 `mean` / `max` / `max_abs` / `last`
  只决定 tfevents 里 `Episode_Metrics/*` 的 **episode 汇总口径**，不影响 CSV。
  两处混用是这份数据最常见的误读来源。
- 本文引用的实测值全部来自一份真实录制：
  `metrics_v03_fwd_t6-fix.csv`（Pikachu v031 T6 档 `model_4999`，前进命令，200 帧 = 4.0 s，
  225 列）。**不同档位/机型/时长的绝对值会变，判读口径不变。**

---

## 2. 两路数据源（理解一切的前提）

这 122 列由**两条互不相通的管线**产出，它们的可信度、参考系、失效模式都不同：

| | 传感器路径 | taxel 几何路径 |
|---|---|---|
| 数据源 | MuJoCo 接触传感器 `feet_ground_contact`（`reduce="netforce"`） | 25 个虚拟 taxel／脚（5 根脚掌胶囊 × 5 点）到地面平面的距离 |
| 覆盖列 | `foot_contact`、`foot_phase`、`foot_duty`、`foot_stance/air_time_s`、`foot_force_*`、`foot_impact_*`、`foot_friction_*`、滑移系列、`base_roll*` | `foot_penetration_m`、`foot_contact_area_m2`、`foot_contact_fraction`、`foot_sensor_count`、`foot_pressure_pa`、`foot_tilt/pitch/roll_deg`、`foot_touchdown_*`、`foot_cop_*` |
| 性质 | **真值**（接触求解器输出） | **几何代理**（鞋底到地面的距离推算） |
| 参考系 | 世界系（力）/ 无 | 世界系（穿透、姿态）/ 踝 body 系（CoP、支撑裕度） |
| 左右标签 | 按名字搜索 primary，**正确** | 按 `grid.side` 索引 body 数组，**反了**（见 3.1） |
| 失效条件 | 求解器漏接触 | 地面非水平、鞋底压痕超出 2 mm 假设 |

**唯一的汇合点是 `foot_pressure_pa`**：力的**总量**取传感器真值，空间**分布**取 taxel 几何代理。
换言之——压力列的"总和"可信，"每一点"是模型推的。

---

## 3. 判读前提

### 3.1 左右脚索引

当前源码已在 `build_foot_sensor_grid` 中按 `("left_ankle_link", "right_ankle_link")`
显式排序踝 body id，与 taxel 的 `side` 顺序一致。因此**新录制的 taxel 派生列、CoP、
足底姿态、压力和触地姿态均按实际 left/right 对齐**；`foot_pressure_pa` 的力值与几何权重
也属于同一只脚，不再交叉配对。接触传感器、site 滑移和关节扭矩列仍按原有映射，不做数据侧对调。

> 历史记录提示：本节及 [COLUMNS.md](COLUMNS.md) 中的实测统计来自修复前的 CSV。
> 该批数据的 taxel 派生列、CoP、姿态及 touchdown 列左右相反，压力还存在传感器力与对侧
> taxel 几何交叉配对。分析旧 CSV 时需交换这些列的左右含义；新录制不需要交换。

### 3.2 地面被当作高度 `ground_z` 的水平面

`penetration = -(taxel_w_z - ground_z)`、`compression = clamp(SOLE_COMPLIANCE_M - gap, min=0)`
都以**单一常数** `ground_z` 为基准（`_ground_height` 取地形 plane 的 z，平地任务为 0）。
平地成立；**斜坡/崎岖地形上穿透量、接触面积、CoP、支撑裕度都会失真**。
力的量（`foot_force_*`、`foot_friction_*`）不受影响。

### 3.3 `SOLE_COMPLIANCE_M = 0.002`：把刚性脚当 2 mm 柔性鞋垫

这是 taxel 路径**唯一的建模假设**，也是它能把刚体-刚体退化接触变成可积分压力分布的原因：

- 接触判据不是"z 低于地面"，而是"z 距地面小于 **2 mm**"（`compression > 0`）。
- `foot_penetration_m = 0` **只表示没有 taxel 低于地面平面，不等于没有接触**。
- 单 taxel 可算出 ~300 kPa 的"压力"（全脚力集中在一个 taxel 上），物理上不可能——
  别当真实压强读。
- `foot_penetration_m` 接近或超过 2 mm，说明几何代理与真实接触开始分家。

---

## 4. 122 列怎么分组

按**物理量**分 6 组（[COLUMNS.md](COLUMNS.md) 的章节顺序），并给出与 dashboard
标签页的对应关系：

| # | 组 | stem 数 | CSV 列数 | 数据源 | 左右可信？ | dashboard 标签页 |
|---|---|---|---|---|---|---|
| A | 步态相位与计时 | 5 | 10 | 传感器 | ✅ | 步态相位 |
| B | 接触力与冲击 | 5 | 10 | 传感器 | ✅ | 足底接触力 |
| C | 接触几何与压力 | 5 | 58 | taxel | ❌ 反 | 接触面积 / 压力分布 |
| D | 足底姿态 | 6 | 12 | taxel | ❌ 反 | 足底姿态 |
| E | CoP 与摩擦 | 6 | 9 | taxel(CoP) + 传感器(摩擦) | ⚠️ 部分反 | CoP 支撑边界 / 足底摩擦 |
| F | 滑移与整机 | 9 | 23 | 传感器 + site + 关节 | ✅ | 足底滑移 / 足部其他 |
| | **合计** | **36** | **122** | | | |

> **36 是 stem 数，不是列数。** 实际 122 列：`foot_pressure_pa` 一个 stem 占 50 列、
> `joint_torque_rate_nm_s` 一个 stem 占 10 列（每个驱动关节一列）。
> 另注：模块 docstring 里写的"约 99 列"是**旧的低估**，与实际的 119 列不符。

另有 3 个"编外"列不在这 6 组内：`base_roll_deg`、`base_roll_rate_rad_s`
（机体横滚，随足底一起注册）与 `joint_torque_rate_nm_s`（关节冲击负载）——
它们描述的是整机而非足底，但同批录制，一并归入 F 组。

---

## 5. 每一组该用什么形式展示

选型的依据是**数据的形态**，不是"看起来像什么图"。同一组里形态不同的列要分开画。

| 组 | 数据形态 | 推荐形式 | 关键视觉元素 |
|---|---|---|---|
| A 步态 | 0/1 区间事件 + 环绕锯齿 + 进行中计时器 | **甘特式区间条**（左右两行）+ 共享 x 轴的相位锯齿带 | 触地色块、换脚标记线、相位 0.5 参考线 |
| B 接触力 | 准方波脉冲 + 带符号冲击尖峰 + 每步锁存 | **左右堆叠面积**（堆叠顶 = 总支持力）+ mg 参考线；rate 另起竖干图 | mg 水平线、切向力虚线（不堆叠）、0 基线 |
| C 几何/压力 | 同一计数的三种换算 + 二维场 | 只画一列（占比填充面积）；压力用**热力图**（时间×taxel + 5×5 足形） | 2 mm 参考线、p99 色标截断、足形网格 |
| D 姿态 | 连续有符号角 + 每次触地锁存 | 三线共享时间轴（pitch/roll 绕 0）+ **每步事件点图** | 0° 参考线、踩平阈值带、触地点标记 |
| E CoP/摩擦 | 足平面二维点 + 裕度标量 + 0~1 比率 | **脚掌平面 xy 散点 + 支撑多边形**；裕度/利用率走时间折线 | 支撑多边形虚线、margin=0 线、util=1 红线 |
| F 滑移/整机 | 门控连续量 + 区间累计 + 离散判据 + 逐关节脉冲 | 逐脚折线 + 接触阴影；离散量用阶梯带；14 条关节冲击用**峰值柱/热力图** | 0.2 m/s 阈值线、0 参考线、左右分色 |

**三条跨组规则**：

1. **必须共享同一根时间轴**：所有时序量用同一个时间基准与缩放控件。
   第 A 组的触地区间应当作为其它所有时序图的**底衬**——`slip=0`、`util=0`、`margin=NaN`、
   `area=0`、`force=0` 全都可能是"腾空"造成的，不叠接触条必然误读。
2. **量纲不同不得共 y 轴**：N 与 N/s、deg 与 rad/s、m 与 m/s 必须分图或分轴。
3. **锁存/区间量不与瞬时量混画**：`foot_impact_peak_n`、`foot_impact_rise_n_s`、
   `foot_touchdown_*`、`foot_contact_slip_dist_m`、`foot_slip_speed_peak_m_s`、
   `foot_duty` 都是"本次支撑内"的量，应画成每步事件图（x = 触地序号或时刻）或明确样式的阶梯。

展开说明与反例见 [DISPLAY.md](DISPLAY.md)。

---

## 6. 判读速查

| 要判什么 | 看哪列 | 阈值/参考 |
|---|---|---|
| 承重是否正常 | `foot_force_normal_n/{l,r}` 之和 | ≈ mg = 6.76 kg × 9.81 ≈ **66.4 N**（本录实测 68.17 N） |
| 是否腾空 | `foot_contact` | 0 = 腾空；本录占空比 **56%** |
| 步态是否对称 | `foot_contact`、`foot_duty` | 左右 mean 应接近；本录均 0.56 / 0.536 |
| 是否拖脚 | `foot_slip_speed_peak_m_s`、`foot_slide_fraction` | **0.2 m/s** 为打滑判据 |
| 是否接近打滑 | `foot_friction_util` | **1.0** = 摩擦极限；本录左脚峰值 1.0000 已触限 |
| 支撑是否将翻 | `foot_cop_margin_m` | 0 = CoP 贴接触区边缘；本录左中位 21 mm、右中位 ≈0.03 mm |
| 落地是否踩平 | `foot_touchdown_tilt_deg` | > 20° = 没放平就砸下去；本录左 ≈21°、右 ≈16° |
| 落地冲击多硬 | `foot_impact_rise_n_s` | 本录 p50 3.1–4.2 kN/s，无绝对阈值，看趋势与左右对称 |
| 脚陷进地面多深 | `foot_penetration_m` | **2 mm**（鞋垫厚）= 代理开始失真；本录 max 5.08 mm |
| 机体是否倾斜 | `base_roll_deg` | 本录 ±6°，均值 ≈0 → 无系统性倾斜 |
| 哪个关节被冲击 | `joint_torque_rate_nm_s/*` | 对列取 max；本录膝/髋 pitch 峰值最高（170–191 N·m/s） |

**四个"0 不是 0"的坑**：

- `foot_penetration_m = 0`：没有 taxel 低于地面平面，**不是**没有接触。
- `foot_contact_slip_speed_m_s = 0`：脚静止**或**脚腾空（乘了 `contact`）。
- `foot_friction_margin_n = 0` / `foot_friction_util = 0`：悬空帧被 guard 成 0，
  看起来像"濒临打滑/离打滑很远"，实际是**没接触**。
- `foot_cop_*` 的 NaN：无接触帧没有 CoP（左 85/200、右 83/200 帧），**不是**数据缺失。

---

## 7. 与 dashboard 的对应

`scripts/plot_logs/metrics_csv_dashboard.html` 把足底列收在导航栏的**一个大项
「足端状态」**下（不再铺 10 个并列标签页），大项内用**二级小标签**切换 6 个组
（`FOOT_SUBGROUPS`，见 `:730`），每组按上表配了专用图型：

| 二级小标签 | 对应本文组 | 图型（一个子标签可挂多张图） |
|---|---|---|
| 步态相位与计时 | A | 触地条带（区间事件）+ 相位锯齿（双 grid 共享 x）；支撑/腾空计时器锯齿；占空比阶梯；每步分布箱线图 |
| 接触力与冲击 | B | 左右法向力堆叠面积 + mg 参考线；力变化率竖干脉冲（独立 y 轴）；每步冲击事件图（峰值/加载率双 grid）；每步分布箱线图 |
| 接触几何与压力 | C | 穿透折线 + 2 mm 参考线；接触占比阶梯面积；压力热力图（时间×taxel 与 5×5 足形两张，腾空段涂灰）；逐脚几何汇总表 |
| 足底姿态 | D | 姿态角折线 + 落地点叠加；每步落地角点图；`sin²` 残差自检图；每步分布箱线图 |
| CoP 与摩擦 | E | 脚掌平面 CoP 轨迹 + 支撑多边形；支撑裕度折线；摩擦利用率/裕度双轴 + μ=1 阈值线；μ 统计磁贴 |
| 滑移与整机 | F | 接触滑移速度折线 + 0.2 m/s 阈值线；每步滑移柱（峰值/累计双 grid）；整机滑移速度 + 滑移脚占比；整机姿态三小图；关节扭矩变化率分组柱；每步分布箱线图 |

每张图都必须在容器上有**显式 CSS 高度**（`#foot-panel > div[id$="-chart"]` 给 260 px
基准，热力图/CoP/三格姿态图等逐个覆盖）。ECharts 按容器 `clientHeight` 建画布，而渲染
函数开头会 `el.innerHTML=''` 清掉占位符——没有高度规则，div 就塌成 0 高，画出来是空图。

三条贯穿全组的显示铁律（详见 [DISPLAY.md](DISPLAY.md)）：

1. **所有时序图共享同一根时间轴，并铺触地区间背景遮罩** —— 腾空期的
   `slip=0 / util=0 / margin=NaN / area=0 / force=0` 全是假值，不铺遮罩必然误读。
   热力图/二维场是唯一例外：整幅铺遮罩会盖住真数据，改用补集 `footAirborneMask()`
   把**腾空**段涂灰，并按 taxel 行分脚（灰带不横穿另一只脚）。
2. **绝不跨量纲共 y 轴** —— N 与 N/s 差约 50 倍、deg 与 rad/s 不同纲，量纲不同就拆
   grid 或拆左右双轴。
3. **锁存量/区间量走「每步事件图」**（x = 触地时刻），不混进逐帧折线语义。
   对应地，**逐帧箱线图只对连续量画**：二值（`foot_contact`）、环绕（`foot_phase`）、
   进行中计时器（`foot_stance/air_time_s`）、离散计数（`foot_sensor_count` 等）、
   二维场（`foot_pressure_pa`）、常量（`foot_ground_friction_mu`）、零膨胀
   （`foot_force_*`/`foot_penetration_m`/`foot_friction_*`/滑移系列）与锁存量
   （`foot_impact_*`/`foot_touchdown_*`/`foot_duty`/`foot_slip_speed_peak_m_s`/
   `foot_contact_slip_dist_m`）都不画——箱体会退化，或概括的根本不是要看的量。
   锁存量的分布改由「每步分布」箱线图表达（`footStepValues` + `renderFootStepBox`）。

> **历史数据注意**：当前源码已修复 taxel 与踝 body 的 left/right 映射，dashboard 按列名直接显示。
> 修复前录制的旧 CSV 仍保留左右反向和 pressure/touchdown 交叉配对；如分析旧文件，须按 3.1 节
> 交换相关列的物理脚含义。新录制的数据无需补偿。

---

## 附：复现本文的实测数字

```bash
# 录制（得到 200 帧 × 225 列）
MUJOCO_GL=egl uv run python scripts/record_v03_metrics.py \
    --checkpoint logs/rsl_ll/pikachu_v031_velocity/<run>/model_4999.pt \
    --out /tmp/footrec --tag t6-fix --clips fwd --seconds 4

# 看图
uv run python scripts/plot_logs/serve.py   # 浏览器打开 metrics_csv_dashboard.html
```
