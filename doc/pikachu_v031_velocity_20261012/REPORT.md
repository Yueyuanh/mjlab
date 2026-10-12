# Pikachu V031 velocity 新 T6 档评测报告（run `2026-10-12_20-05-34`）

- **被评对象**：`mjlab/logs/rsl_rl/pikachu_v031_velocity/2026-10-12_20-05-34`
  （任务 `Mjlab-Velocity-Flat-Pikachu-V031-T6`，4096 env × 5000 iter，训练 seed 42，
  主机 `finnox-Legion-R7000-APH9` / RTX 4060 Laptop）。
- **参照物**：`doc/pikachu_v031_velocity_20261010/` 那份报告里的旧 T6 run
  （`2026-10-10_20-11-53`，主机 `finnox-4090` / RTX 4090）。
- **评测协议**：`scripts/eval_v031_symmetry.py`（7 条命令 × 2 个 seed）、
  `scripts/record_v03_metrics.py`（受控回放，fwd +0.40 m/s，8 s）、
  `scripts/diag_v031_rollout.py`（12 s 基座姿态遥测，fwd，seed 42）。
- **本目录产物**：`analyze_run.py`、`summary.json`、
  `training_selected_scalars.csv`、`metrics/`、`symmetry/`、`diagnostics/`、
  `plots/`、`videos/`。生成方式见文末「复现」。

> 口径说明：`analyze_run.py` 与 20261010 那版的列契约、`JOINT_LIMIT = 0.26`、
> 移动平均窗口、tfevents 读法逐字一致，所以两目录里的同名数字可以直接对着看。

---

## 结论摘要

1. **这次训练本身是健康的**：5000/5000 跑完，`mean_reward` 从 −1.90 升到 48.33
   （末 1000 迭代均值 48.62），`mean_episode_length` 980.2/1000，策略 `mean_std`
   从 0.99 收到 0.31，追踪奖励稳定在 1.22，没有任何崩溃或 NaN。

2. **但视觉上那个现象在新的 T6 上原样复现，而且比旧 run 更"整齐"**。
   7 条验收线里有 4 条不达标，失败集中在**需要换步的平移命令**上；
   `stand` / `turnL` / `turnR` 在所有 7 个 checkpoint 上都是干净的。

3. **不是模型配置问题**（这一条本次又复核了一遍）：两次 run 的
   `params/env.yaml` **逐字节相同**（md5 `bd9a50d8a46e5cad04a107e3d115801e`），
   也就是奖励权重/参数、观测、事件、资产引用**全部一致**；tfevents 的 tag 集合
   也完全一致（各 158 个）；seed 都是 42；`clip_actions` 都是 `null`；
   `agent.yaml` 只差一个 `max_iterations`（它只是 `train.py:198` 的循环上界）
   和一个 `symmetry_cfg: null`（即对称扩展在两份里都是关的）。
   **两次 run 用的是同一版资产**：质量修复 `044959e` 在 2026-10-12 **21:41** 才落地，
   晚于新 run 启动的 20:05（新 run 启动时保存的 `git/mjlab.diff` 里没有 xml），
   所以两次都跑在修复前的 `4211929c…` 上。当前 HEAD 的资产已是修复后的
   `d68be1ad…`——**这正是下一批消融应该用的版本**。唯一的实质差异是**机器**。

4. **根因指向奖励函数/动作空间，不是资产**。三条跨 7 个 checkpoint（两次独立训练、
   两台机器）都稳定的窄带信号：
   - **hip_roll/hip_yaw 目标长期停在 ±0.26 限位之外**——越界率 85%–92%；
     训练日志里 hip_roll 目标的 episode `max|t|` 末 1000 迭代均值是
     **左 0.943 / 右 0.974 rad**，是限位的 3.6 倍，而且**每一个迭代**都越界。
   - **共模偏置** `hr_d = (qL + qR)/2` 稳定偏 **0.057–0.096 rad**（验收线 0.03）。
   - **镜像等价性** `mirror_equiv` 只有 **0.70–0.78**（1.0 才是完美镜像）。
   奖励里没有任何一项直接惩罚"目标越界"：`dof_pos_limits` 罚的是**关节位置**，
   而位置被软限位挡住（末 1000 迭代 `max|q|` 左 0.271 / 右 0.266，刚好骑在 0.26 上）；
   `standing_posture_l2` 只在命令 ≈0 的环境生效，走路环境恒为 0。

5. **建议**：剩余四档（Clip / Sym / SymClip / YawHuber）值得继续，但**判据换成
   上面那三条窄带指标**，`heading_drift` 只当定性参考——它在单个 run 内部就能波动
   5.8 倍（本次 46.8° → 29.9° → 25.9°，上次 91.6° → 21.2° → 123.7° → 75.7°）。

---

## 数据与视频

### 1. 训练全程趋势

![training curves](plots/training_curves.png)

数据源：本 run 保存的 `events.out.tfevents.1791806737.finnox-Legion-R7000-APH9.1183624.0`
（158 个 tag，与旧 run 的 tag 集合完全一致）。抽取的 14 条曲线落在
`training_selected_scalars.csv`，逐迭代 5000 行。

| 指标 | @0 | @1000 | @3000 | @4999 | 末 1000 均值 |
|---|---|---|---|---|---|
| `Train/mean_reward` | −1.90 | 50.01 | 45.65 | 48.33 | 48.62 |
| `Train/mean_episode_length` | 17.6 | 972.1 | 963.7 | 980.2 | 981.6 |
| `Policy/mean_std` | 0.993 | 0.301 | 0.318 | 0.310 | 0.310 |
| `Episode_Reward/track_linear_velocity` | 0.003 | 1.266 | 1.168 | 1.221 | 1.221 |
| `Episode_Reward/track_angular_velocity` | 0.0001 | 0.075 | 0.068 | 0.081 | 0.081 |
| `Episode_Metrics/vel/err_yaw` | +0.057 | −0.005 | +0.013 | −0.002 | +0.001 |

**关于 hip_roll，训练日志本身就是证据**（这是本次新挖出来的一列，旧报告没用到）：

| 量（episode `max|·|`） | 末 1000 迭代均值 | 越界（>0.26）的迭代占比 |
|---|---|---|---|
| `joint_position_rad/left_hip_roll_joint` | 0.2707 | 95.0% |
| `joint_position_rad/right_hip_roll_joint` | 0.2659 | 81.0% |
| `joint_target_rad/left_hip_roll_joint` | **0.9434** | **100%** |
| `joint_target_rad/right_hip_roll_joint` | **0.9737** | **100%** |

两点结论：

- **目标越界是训练全程的常态，不是某个 checkpoint 的偶发**：5000 个迭代里，
  hip_roll 目标的 episode 峰值**每一次**都超过 0.26。
- **两侧目标的幅值是近似对称的**（0.943 vs 0.974，差 0.030 rad）。所以不对称不在
  "哪条腿摆得大"，而在**符号/相位**——两侧同时往同一边偏，正是 `hr_d` 测的那个量。
  这解释了为什么"看起来一条腿歪"其实是整体在往一侧压。

### 2. 受控回放

![controlled replay comparison](plots/controlled_replay_comparison.png)

条件：档位固定 `fwd`（命令 +0.40 m/s），每档 8 s，seed 42，1 个 env。
`nominal` = 关观测噪声 + 关推力 + **弹掉启动 DR**（摩擦/编码器/COM）；
`DR` = 关观测噪声 + 关推力，保留启动 DR；`train cond.` = 全训练条件。
原始 CSV 在 `metrics/model_<ck>_<mode>/`。

| 条件 | 实测 vx | 净位移 | yaw RMS | \|hr_d\|max | p2p 比 | 越界 | L 摆幅 | R 摆幅 |
|---|---|---|---|---|---|---|---|---|
| m1000 nominal | 0.38 | 2.92 m | 0.960 | 0.181 | 0.821 | 92.8% | 0.186 | 0.227 |
| m1000 DR | 0.39 | 2.86 m | 1.041 | 0.199 | 0.797 | 91.0% | 0.198 | 0.248 |
| m1000 train cond. | 0.39 | 2.36 m | 1.052 | 0.194 | **1.704** | 85.2% | **0.521** | 0.305 |
| m4999 nominal | 0.40 | 3.00 m | 1.043 | 0.176 | 0.812 | 84.8% | 0.200 | 0.246 |
| m4999 DR | 0.39 | 3.06 m | 1.060 | 0.183 | 0.900 | 91.2% | 0.204 | 0.227 |
| m4999 train cond. | 0.38 | 2.54 m | 0.976 | 0.216 | 0.641 | 79.8% | 0.296 | **0.461** |

- **前进追踪没问题**：6 个条件的实测 vx 都是 0.38–0.40（命令 0.40），8 s 净位移
  2.36–3.06 m。所以"走不动"不是这次的问题。
- **yaw RMS 0.96–1.06 rad/s，是验收线 0.25 的 4 倍**，而且三种 DR 条件下都在
  ——说明它不来自摩擦/编码器/质心的随机化，是策略本身的行为。
- **`hr_d` 和越界率在三种条件下几乎不变**（0.176–0.216 / 79.8%–92.8%），
  同样指向"策略自带的结构性偏置"，而不是被域随机化激发出来的。
- **`train cond.` 那一列的两个离群值（p2p 1.704、L 摆幅 0.521 / R 摆幅 0.461）
  不要当结论**：它只跑了 1 个 env、1 个 seed、8 s，噪声和推力在这个样本量下
  统计意义很弱，单次抽到的噪声实现就能把摆幅推大。要下结论得加 env 数。

![replay diagnostics](plots/replay_diagnostics.png)

上图为 `m4999 DR` 那一段的逐帧遥测（视频：`videos/controlled_replay_telemetry.mp4`，
8 s，50 fps）。注意右上"Unclipped policy targets"：两条目标曲线在 ±0.26 虚线外
长期停留，而不是偶尔探出去——这就是越界率 91% 的来源。

### 3. 基座姿态遥测（回放 CSV 里没有 roll/pitch，这一段单独跑）

`diagnostics/newT6_model_{1000,4999}_rollout_diag.json`，fwd 12 s，seed 42：

| 量 | model_1000 | model_4999 |
|---|---|---|
| `roll_mean` | −0.0049 | −0.0003 |
| `roll_std` | 0.0718 | 0.0753 |
| `pitch_mean` | −0.0042 | +0.0203 |
| `yaw_drift` | +1.295 rad（+74.2°） | −0.637 rad（−36.5°） |
| `yaw_rate_rms` | 1.457 | 1.430 |
| `pos_y_drift` | +2.550 m | −2.388 m |
| `pos_x_travel` | +3.122 m | +3.941 m |
| `qL_mean` / `qR_mean` | +0.1607 / +0.0155 | +0.1598 / −0.0216 |
| `qL_std` / `qR_std` | 0.0627 / 0.0918 | 0.0640 / 0.0837 |
| `qL_max` / `qR_max` | +0.2731 / +0.1399 | +0.2715 / +0.0961 |
| 共模 `(qL+qR)/2` | +0.0881 ± 0.0748 | +0.0691 ± 0.0722 |
| 差模 `(qL−qR)/2` | +0.0726 | +0.0907 |
| raw action L / R | −0.379 / **+1.057** | −0.203 / **+0.811** |

三件事：

- **基座本身没有静态倾斜**：`roll_mean` 是 −0.005 / −0.000 rad。所以视觉上
  "向一侧倾斜"不是躯干歪，而是**步态在往一侧画弧**（12 s 里横漂 2.4–2.6 m）。
- **共模偏置在两次独立跑里都是 0.069–0.088 rad，且抖动量级相同（std 0.072–0.075）**
  ——和第 1 节训练日志、第 4 节验收线三处独立数据源给出同一个数。
- **raw action 的左右均值差极大**（−0.20 vs +0.81，即差 1.0 个 action 单位 =
  0.5 rad 的目标偏移）。这是最直白的"策略在用一个恒定的同向偏置补偿什么"的证据。

### 4. 验收线对账

![acceptance reconciliation](plots/symmetry_acceptance.png)

7 个 checkpoint 对 4 条验收线（蓝 = 本次 run，灰 = 2026-10-10 run）：

| checkpoint | fwd Δyaw | fwd 越界率 | 最坏 \|hr_d\| | fwd p2p 比 | `mirror_equiv` | 摔倒 |
|---|---|---|---|---|---|---|
| **newT6 m1000** | 46.8° | 90.9% | 0.0868 | 0.681 | 0.729 | 否 |
| **newT6 m3000** | 29.9° | 87.4% | 0.0962 | 0.557 | 0.704 | 否 |
| **newT6 m4999** | 25.9° | 91.5% | 0.0681 | 0.763 | 0.708 | 否 |
| oldT6 m1000 | 91.6° | 85.5% | 0.0654 | 0.713 | 0.714 | 否 |
| oldT6 m3000 | 21.2° | 88.1% | 0.0770 | 0.704 | 0.754 | 否 |
| oldT6 m5000 | 123.7° | 85.0% | 0.0765 | 0.686 | 0.755 | 否 |
| oldT6 m29999 | 75.7° | 89.0% | 0.0569 | 0.850 | 0.781 | 否 |

| 验收线 | 阈值 | newT6 m4999 | 判定 |
|---|---|---|---|
| hip_roll/hip_yaw 目标越界率 | 0% | 91.5% | ✗ |
| 8 s 直行航向漂移 | <10° | 25.9° | ✗ |
| 零偏航命令 yaw RMS | <0.25 rad/s | 1.060（fwd 回放） | ✗ |
| 两侧 \|q\| 均值差 | <0.03 rad | 0.068 | ✗ |
| hip_roll p2p 比 | 0.8–1.25 | 0.763 | ✗（贴边） |
| 20 s 标称成功率 | >99% | 12 s 无摔倒、`episode_length` 980/1000 | ✓（口径不同，见「局限」） |
| 20 s 扰动成功率 | >95% | 同上 | ✓（口径不同） |

**没有一个 checkpoint 过线，而且不是"这次特别差"**：旧 run 的 4 个 checkpoint 在
越界率（85.0–89.0%）、`|hr_d|`（0.0569–0.0770）、`mirror_equiv`（0.714–0.781）
这三条上落在**和本次同一个区间**里。两次独立训练、两台机器、7 个 checkpoint，
这三个量都稳——它们是可做判据的；`Δyaw` 不是。

`mirror_equiv` 的含义：把同一段观测左右镜像后喂给策略，比较输出的动作差
（`mean_abs_diff / act_std`）。0.70–0.78 意味着**镜像后的动作差达到了动作标准差的
七成**——策略对"左右互换"几乎完全不敏感，这是不对称的直接度量，且不依赖任何阈值。

---

## 模型配置检查

### 已排除的明显左右配置错误（沿用 20261010 报告的结论，本次复核未变）

- URDF/MJCF 的左右连杆质量、惯量、几何在数值上逐项对称（`diag_v031_symmetry.py`）。
- 关节轴、限位、传动比左右一致；`left_knee_joint` / `right_knee_joint` 的命名后缀
  在 `placo_defaults.json` 与 `Entity.joint_names` 之间一致。
- 曾经存在的 `left_hip_yaw_link` 与右侧差 10.3 g 的质量不对称，已在
  `044959e` 修成两侧都是 0.333676 kg；剩下的 0.58 mm 质心镜像残差来自
  `hip_yaw_link` 的网格几何本身，已在 `9b288e5` 拆解清楚，量级不足以产生 0.07 rad
  的 `hr_d`。
  ⚠️ **本报告评的这一次 run 跑在修复之前**（`044959e` 落地于 2026-10-12 21:41，
  run 启动于 20:05），所以下面的结论是"在没有质量不对称修复的条件下"得出的。
  这反而更有说服力：**修复前后 `hr_d` 都在 0.057–0.096 rad 这个区间**（旧 run 也
  跑在修复前），说明这 10.3 g 不是 `hr_d` 的来源。

### 本次新增的跨 run 复核

| 项 | 旧 run `2026-10-10_20-11-53` | 新 run `2026-10-12_20-05-34` | 是否相同 |
|---|---|---|---|
| `params/env.yaml` md5 | `bd9a50d8a46e5cad04a107e3d115801e` | 同左 | ✓ **逐字节** |
| tfevents tag 集合 | 158 个 | 158 个 | ✓ 集合相等 |
| 资产 `pikachu_v031_train.xml` | `4211929c…b56ee7`（修复前） | 同左（修复前） | ✓ |
| seed | 42 | 42 | ✓ |
| `clip_actions` | `null` | `null` | ✓ |
| `agent.yaml` 差异 | — | `max_iterations` 30000→5000、多一个 `symmetry_cfg: null` | 均惰性 |
| 启动时未提交的代码 | 34 个文件（基线提交已从对象库消失） | 4 个文件（消融脚手架） | 见下 |
| 主机 / GPU | `finnox-4090` / RTX 4090 | `finnox-Legion-R7000-APH9` / RTX 4060 Laptop | ✗ |

两点要讲清楚：

- **`env.yaml` 逐字节相同是最强的一条**：它序列化了整个 EnvCfg，奖励项的权重与
  `params`、观测项、事件、命令范围、资产引用全在里面。这一条相等，就说明
  **两次训练的环境配置没有任何数值差异**。
- **"代码逐行相同"不能断言，但差异是惰性的**。新 run 启动时（20:05）工作区比
  `4fb5e18` 多了 4 个文件的改动：`rl/config.py` 加了一个默认 `None` 的
  `symmetry_cfg` 字段、`velocity/mdp/rewards.py` 加了一个尚未被任何奖励项引用的
  `angular_velocity_huber_error`、`pikachu_v031/__init__.py` 与 `rl_cfg.py` 把注册从
  1 档扩到 5 档。对 T6 这一档来说这四处全部不生效（`symmetry_cfg: null`、
  `clip_actions: null`、T6 的 env cfg 一字未改，`env.yaml` 相同就是证据）。
  旧 run 启动时工作区另有 34 个文件的改动，但其基线提交已从对象库消失，
  无法逐行重建——所以只能说**配置确证相同，代码不能逐行对账**。

**结论：现象不是某次训练的偶发，也不是配置差异造成的；换机器、换 run 都复现。**

### 为什么"目标越界"能长期存在而没被惩罚

奖励项里跟关节边界有关的只有三条，都不直接约束**目标**：

- `dof_pos_limits`（`Episode_Reward/dof_pos_limits` 末 1000 迭代均值 −0.0004）：
  罚的是**关节位置**偏离限位，位置被 MuJoCo 的软限位挡住，代价很小。
- `torque_soft_limit`（−0.0054）：罚力矩超软限，与目标位置无关。
- `standing_posture_l2`：只在命令 ≈0 的环境生效（`command_name="twist"`），
  走路环境恒为 0。

而动作项 `JointPositionActionCfg(scale=0.5, use_default_offset=True)` 把策略的
raw action 直接映射成目标：raw action 到 ±4.7（`action_absmax` 实测值）时目标就到
±2.3 rad，远超 0.26。**策略没有任何梯度压力把 raw action 收回来**——这正是 Clip 档
（±1.0 action clip）要对准的东西。

---

## 改进方向与实验顺序

五档单因素消融已经在代码里（`src/mjlab/tasks/velocity/config/pikachu_v031/__init__.py`），
每档对准一条窄带指标：

| 档 | 任务名 | 改的东西 | 对准的指标 | 优先级 |
|---|---|---|---|---|
| T6 | `Mjlab-Velocity-Flat-Pikachu-V031-T6` | 基线 | — | 已完成 |
| Clip | `…-V031-Clip` | action clip ±1.0 | 越界率 91% → ? | **P0，先跑** |
| Sym | `…-V031-Sym` | 镜像增广 + mirror loss | `mirror_equiv` 0.71 / `hr_d` 0.068 | **P0，先跑** |
| SymClip | `…-V031-SymClip` | 两者叠加 | 上面两条一起 | P1，看前两档结果 |
| YawHuber | `…-V031-YawHuber` | 偏航 Huber 奖励 | yaw RMS 1.06 | P1 |

**顺序理由**：Clip 和 Sym 各自对准一条**互不重叠**的窄带指标，能干净地做单因素归因；
SymClip 是它们的交互项，必须在两者都有结果之后才有信息量；YawHuber 压的是
`Δyaw` 这条高噪声指标，放最后、且只作定性判据。

**判据（替代 `Δyaw`）**：
1. 越界率（`eval_v031_symmetry.py` 的 `target_violation`，跨 checkpoint 稳定）；
2. `|hr_d|` 最坏值（跨命令，验收线 0.03）；
3. `mirror_equiv`（验收线 → 1.0）；
4. `hr_p2p_ratio`（验收线 0.8–1.25）。
`Δyaw` / `heading_drift_deg_per_m` 只用来回答"有没有走成弧线"。

---

## 实际执行的实验矩阵

| 档 | 任务名 | 状态 | 进度 | 最后 `mean_reward` | checkpoint |
|---|---|---|---|---|---|
| T6（本次） | `…-V031-T6` | **完成** rc=0 | 5000/5000 | 48.33 | `model_0/500/…/4500/4999.pt`（11 个） |
| Sym | `…-V031-Sym` | **已按指令暂停** | 898/5000 | 37.30 | `model_0.pt`, `model_500.pt` |
| Clip | `…-V031-Clip` | 未启动 | — | — | — |
| SymClip | `…-V031-SymClip` | 未启动 | — | — | — |
| YawHuber | `…-V031-YawHuber` | 未启动 | — | — | — |

- 四份日志（`/tmp/v031_ablation/*.log`）里 **Traceback / CUDA out of memory /
  RuntimeError 命中数都是 0**；T6 正常退出；GPU 空闲（438 MiB / 0%）。
- Sym 档是被我按用户指令主动停的，不是崩溃，**不需要重启**。
- 若要继续，建议**重开一批统一的五档**（都用修复后的资产），而不是接着 Sym 那
  898 个迭代跑——否则五档的起点不一致，单因素归因就不干净了。

---

## 本轮实验的已知局限

1. **单训练 seed**：五档各只有 1 个训练 seed（42），评测 2 个 seed（42, 7）。
   窄带指标跨 checkpoint 稳，但没有跨训练 seed 的方差估计。
2. **跨机器不可配对**：两次 T6 run 分别跑在 RTX 4090 和 RTX 4060 Laptop 上。
   第 4 节的逐项核对说明**配置**相同，但不同 GPU 上的浮点累加顺序、cuDNN 算法选择、
   以及不同的 wall-clock 都会让 PPO 轨迹分叉。所以两列数字可以并排看区间，
   不能做"A 比 B 好 X%"的配对比较。
3. **`heading_drift` 噪声极大**：单个 run 内部就波动 5.8 倍（本次 46.8° → 29.9°
   → 25.9°；上次 91.6° → 21.2° → 123.7° → 75.7°）。它只能定性用。
4. **受控回放的样本量**：每档 8 s、1 个 env、1 个 seed。`--mode train` 只有 1 个 env，
   噪声/推力的统计意义很弱——第 2 节表里 `m1000 train cond.` 的 p2p 1.704 就是这个
   原因，**不能读成"训练条件下左腿摆幅翻倍"**。要下这个结论得加 env 数重录。
5. **回放 CSV 不含基座 roll/pitch**（`record_v03_metrics.py` 的列契约里没有），
   所以姿态结论全部来自 `diag_v031_rollout.py` 的 JSON，两套数据的 seed 与时长
   （8 s vs 12 s）不完全对齐。
6. **验收线里的两条"成功率"口径不同**：原验收线要求 20 s 的标称/扰动成功率，
   本次用的是 `mean_episode_length`（训练期，20 s 上限，980/1000）与 12 s 无摔倒
   两个代理量。严格达标需要在 `eval_v031_symmetry.py` 里把 `--seconds` 提到 20。

---

## 复现

全部从 `mjlab/` 目录下执行。

```bash
# 0) 训练（已完成，此处仅为记录；无头、后台、必须指定 GPU）
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 WANDB_MODE=disabled
uv run train Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --env.scene.num-envs 4096 --agent.max-iterations 5000 --agent.save-interval 500 \
  --log-root logs/rsl_rl
# 等价的一行封装（五档共用同一套预算，保证同预算对照）：
#   bash scripts/run_v031_arm.sh Mjlab-Velocity-Flat-Pikachu-V031-T6

# 1) 协议化评测（7 条命令 × 2 seed）→ symmetry/
uv run python scripts/eval_v031_symmetry.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --checkpoint logs/rsl_rl/pikachu_v031_velocity/2026-10-12_20-05-34/model_4999.pt \
  --out doc/pikachu_v031_velocity_20261012/symmetry/newT6_model_4999.json

# 2) 受控回放（fwd，8 s，三种随机化条件）→ metrics/model_<ck>_<mode>/
for MODE in nominal dr train; do
  uv run python scripts/record_v03_metrics.py \
    --task Mjlab-Velocity-Flat-Pikachu-V031-T6 \
    --checkpoint logs/rsl_rl/pikachu_v031_velocity/2026-10-12_20-05-34/model_4999.pt \
    --clips stand,fwd --seconds 8 --seed 42 --mode "$MODE" \
    --tag "model_4999_$MODE" \
    --out "doc/pikachu_v031_velocity_20261012/metrics/model_4999_$MODE"
done

# 3) 基座姿态遥测（回放 CSV 里没有 roll/pitch）→ diagnostics/
uv run python scripts/diag_v031_rollout.py \
  --checkpoint logs/rsl_rl/pikachu_v031_velocity/2026-10-12_20-05-34/model_4999.pt \
  --seconds 12 --seeds 42 --out /tmp/v031_diag_new/model_4999

# 4) 汇总出图/表/视频 → plots/ videos/ summary.json training_selected_scalars.csv
uv run python doc/pikachu_v031_velocity_20261012/analyze_run.py
```

> 注：`--mode {nominal,dr,train}` 是本轮新加到 `scripts/record_v03_metrics.py` 的开关
> （`nominal` 会额外弹掉 `foot_friction` / `encoder_bias` / `base_com` 三个启动期
> 域随机化事件）。旧目录 `doc/pikachu_v031_velocity_20261010/metrics/` 下的 CSV 是当时
> 用一次性命令录的，命令本身没有记录在仓库里——这是上一版的可复现性缺口，本版补上了。
