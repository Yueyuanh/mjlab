# Pikachu V031 velocity 单侧偏航与 hip_roll 不对称诊断报告

**对象**：`logs/rsl_rl/pikachu_v031_velocity/2026-10-10_20-11-53`  
**任务**：`Mjlab-Velocity-Flat-Pikachu-V031-T6`  
**日期**：2026-10-11

## 结论摘要

结论不是“左侧 MJCF 关节范围配错”。**运动学层面已用 FK 判定性地排除**：
20 组随机驱动关节配置下，左右腿全部 14 个 body 的镜像位置残差
**0.000000000 mm**、姿态残差 **0.000e+00**，左右腿是严格镜像机构；左右
`hip_roll` 的轴、硬限位、PD 参数和扭矩上限逐项相同（详见“模型配置检查”）。
训练 TensorBoard 的最后 1000 轮中，左右 `hip_roll` 的 episode 最大绝对角均值
也仅相差 **0.0068 rad**。

主要问题属于**策略/控制配置与奖励共同造成的非对称饱和策略**：

1. **首要配置问题：动作没有裁剪。** `agent.yaml` 中 `clip_actions: null`，
   `env.yaml` 中关节动作 `clip: null`。`hip_roll` 的动作缩放虽为 0.2 rad，
   但策略输出本身无界，所以位置目标并不受 ±0.2 rad 约束。
2. **训练已学会依赖越界目标产生的饱和扭矩。** 原始异常 rollout 中，左右
   `hip_roll` 目标分别有 30.1% 和 30.7% 的采样越过 ±0.26 rad 硬限位，最大
   达到 −0.469 / +0.603 rad。训练最后 1000 轮的 episode 最大目标均值更达到
   左 0.929、右 0.981 rad，而实际关节最大角仍约为 0.269 / 0.262 rad。
   `dof_pos_limits` 只看实际位置，因而无法约束这种越界目标。
3. **奖励是左右对称的，但没有约束策略必须对称。** PPO 配置没有
   `symmetry_cfg`、镜像数据增强或 mirror loss。对称任务本来就允许多个等价最优
   解；随机初始化、微小 CAD 差异和扰动会让策略选中某个单侧解。现有
   `standing_posture_l2` 只在近零命令时生效，走路时没有 DC 偏置约束。
4. **偏航恢复奖励在大误差处饱和。** `track_angular_velocity` 使用
   `exp(-error² / 0.4²)`；误差为 1 rad/s 时奖励仅约 0.0019，梯度已很弱。
   同时 `body_angular_velocity_penalty` 明确不惩罚 z 轴角速度。因此策略一旦进入
   大偏航状态，缺少持续拉回零偏航的非饱和梯度。
5. **域随机化是触发/放大因素，不是唯一根因。** 训练每 1–3 秒直接注入一次
   速度扰动，横向范围 ±0.5 m/s、偏航范围 ±0.78 rad/s，均大于训练初始命令
   范围；base COM 还可在 x/y 各偏 ±25 mm。受控 seed 42 测试没有复现原 CSV
   的同一条严重轨迹，但这种强扰动会鼓励“顶限位救姿态”的恢复策略。

因此应把问题定性为：**控制配置缺少动作安全边界是根因，奖励缺少非饱和偏航
恢复与镜像约束是第二根因；当前证据不支持左右 hip_roll 机械配置错误。**

**改进实验**：已按“单因素隔离”原则设计并启动五档**同预算**训练
（T6 / Clip / Sym / SymClip / YawHuber，各 4096 env × 5000 iter），每档相对
基线只改一个因素，矩阵与验收线见下文。

**T6 档已训完并评测（2026-10-12）**：现象在新 run 上**稳定复现**，且在
7 个 checkpoint 上失败签名一致——带平移的步态系统性漂移（fwd 8 s 偏航
25.1°/26.8°，越界率 0.915，`|hr_d|` 0.068，p2p 比 0.763），而 stand /
turnL / turnR 全部干净。但旧 run 的 4 个 checkpoint 自身 Δyaw 就在
21–127° 之间波动，说明**单 seed 单 checkpoint 的直行偏航噪声极大、不可用于
2 倍以内的对比**；跨 run 逐项核对（资产 hash、`env.yaml` md5、`agent.yaml`、
seed、未提交代码）后确认两次 run 代码/配置/资产相同，唯一实质差异是机器
（4090 服务器 vs 4060 Laptop），故本轮以 newT6 为基线、不再用旧 run 配对。
详见「新 T6 档评测结果与跨 run 可比性核查」。

**顺带修掉的模型缺陷**：体检中唯一量级超过 1 g 的左右差异是
`left_hip_yaw_link` 比右侧轻 **10.311 g**（3.09%），已按"两侧同值 0.333676 kg"
修复（commit `044959e`，URDF / `pikachu_v031.xml` / `pikachu_v031_train.xml`
三处同步）。它的重力力矩量级只有 hip_yaw 电机 effort_limit 的 0.02%，**不是**
倾斜现象的原因，属于把模型收拾干净。修完第 1 节所有 `dm(g)` 归零、第 5 节
镜像残差仍是 0.000000000 mm。

## 数据与视频

### 1. 原始异常 rollout

![原始 rollout 诊断曲线](plots/provided_rollout_diagnostics.png)

- 命令固定为 `(vx, vy, wz) = (0.30, 0, 0)`。
- 6.9 秒内朝向累计变化 **115.8°**，实际偏航角速度 RMS 为
  **1.011 rad/s**。
- 左 `hip_roll` 均值为 **+0.138 rad**，右侧为 **−0.034 rad**。两侧摆幅
  其实接近（0.217 / 0.212 rad）；用户观察到的“左侧范围更偏”主要是
  **DC 中心偏移**，不是左腿振幅单独增大。
- 左侧实际角达到 +0.267 rad，已经贴近/轻微穿过 +0.26 rad 硬限位。
- 左右位置目标 RMSE 分别为 0.268 / 0.241 rad，说明位置执行器长期处于大误差、
  大扭矩补偿状态。

[原始异常同步遥测视频](videos/provided_rollout_telemetry.mp4)

### 2. 训练全程趋势

![训练曲线](plots/training_curves.png)

最后 1000 轮摘要：

| 指标 | 数值 |
|---|---:|
| mean reward | 53.03 |
| mean episode length | 994.58 / 1000 steps |
| policy mean std | 0.289 |
| 左 / 右 hip_roll 实际最大绝对角 | 0.269 / 0.262 rad |
| 左 / 右 hip_roll 目标最大绝对角 | 0.929 / 0.981 rad |
| 左右实际角统计差 | 0.0068 rad |
| `dof_pos_limits` episode reward | −0.00038 |
| `torque_soft_limit` episode reward | −0.00596 |

曲线说明训练看起来已经“高回报、长 episode”，但策略目标早已远超机械行程。
限位奖励几乎为零，只证明物理位置被硬限位挡住，并不表示控制目标健康。

[筛选后的训练标量 CSV](training_selected_scalars.csv)

### 3. 受控回放与消融

![受控回放对比](plots/controlled_replay_comparison.png)

所有回放使用单环境、固定命令和 seed 42。`nominal` 关闭 startup DR、观测噪声
和周期推力；`DR` 保留 startup 摩擦/编码器/COM 随机化；`train cond.` 还恢复
观测噪声和周期推力。

| checkpoint / 条件 | 时长 | 左 / 右 hip_roll 均值 (rad) | 目标越界率 左 / 右 | yaw RMS | 朝向变化 |
|---|---:|---:|---:|---:|---:|
| 原 CSV，m1000，0.30 m/s | 6.9 s | +0.138 / −0.034 | 30.1% / 30.7% | 1.011 | +115.8° |
| m1000，nominal，0.40 m/s | 8.0 s | +0.161 / −0.043 | 26.0% / 31.5% | 1.131 | +91.4° |
| m1000，DR，0.30 m/s | 8.0 s | +0.134 / −0.001 | 26.5% / 38.8% | 1.173 | +134.9° |
| m1000，train cond.，0.30 m/s | 8.0 s | +0.141 / −0.015 | 28.0% / 32.8% | 1.015 | +52.0° |
| m29999，nominal，0.40 m/s | 8.0 s | +0.163 / −0.056 | 19.8% / 32.5% | 0.883 | −76.7° |
| m29999，nominal，raw action clip ±1 | **2.8 s 后摔倒** | +0.175 / −0.083 | 0% / 0% | 0.890 | −134.9° |

主要观察：

- 从 m1000 训练到 m29999 后，前进速度跟踪改善，但 hip_roll 的 DC 不对称和
  大幅偏航没有消失。
- startup DR、观测噪声和推力会改变严重程度，却不是产生 DC 偏置的必要条件；
  nominal 条件下也存在。
- 给**现有权重**临时加 `raw action clip ±1` 虽使目标严格回到 ±0.2 rad，
  但 2.8 秒即摔倒。这证明旧策略依赖越界目标，不能在部署端单独热修，必须从
  训练开始启用裁剪并重训。

最终 checkpoint 标称条件视频：

- [侧视 6 秒](videos/fwd__side__model_29999_nominal.mp4)
- [正视 6 秒](videos/fwd__front__model_29999_nominal.mp4)

原始和受控逐帧 CSV 位于 [`metrics/`](metrics/)，机器可读摘要位于
[`summary.json`](summary.json)。

## 模型配置检查

### 已排除的明显左右配置错误

**运动学（决定性判据）**：`scripts/diag_v031_symmetry.py` 第 5 节逐关节激励
**独立测出**镜像符号表（`hip_pitch / hip_roll / hip_yaw / ankle = -1`、
`knee_up = +1`），与 `symmetry.MIRROR_SIGN` 逐项一致；随后在 20 组随机驱动关节
配置（±0.4 rad）下比较左右腿全部 14 个 body 的世界位姿，**位置残差
0.000000000 mm、姿态残差 0.000e+00**。左右腿是严格镜像机构，运动学层面不存在
任何不对称。

> 这一节此前只有 qpos=0（CAD 零位）下的静态比较，而零位时**所有关节角都是 0**，
> `hip_pitch / hip_roll / hip_yaw` 的符号差异完全体现不出来 —— 静态量查不出
> 运动学问题。非零配置下的 FK 验证才是决定性判据。

**执行器**：左右共用同一组正则（`.*_hip_pitch_joint` 等）与同一组常量派生的
stiffness / damping / armature / effort_limit，编译后 `gainprm / biasprm /
ctrlrange / forcerange` 左右逐项相同（残差 0）。脚底摩擦随机化用
`shared_random: true`，不会主动给左右脚不同摩擦系数。

| 项目 | left hip_roll | right hip_roll |
|---|---:|---:|
| joint axis | `[1, 0, 0]` | `[1, 0, 0]` |
| hard range | `[-0.26, 0.26]` | `[-0.26, 0.26]` |
| action scale | 0.20 | 0.20 |
| stiffness | 8.709 | 8.709 |
| damping | 0.554 | 0.554 |
| effort limit | 12.5 N·m | 12.5 N·m |

**唯一真实的左右不对称**在质量：`hip_yaw_link` 左 0.323365 kg / 右 0.333676 kg，
差 10.311 g；左腿链 1.063113 kg / 右腿链 1.073424 kg，差 0.961%；其余各 link
质量差均为 0，镜像 COM 残差最大 0.58 mm。该差异**直接写在 URDF 的
`<inertial>` 里**（不是 URDF→MJCF 转换引入的），左右件的惯量也同步偏 0.2–0.4%，
形态上像是 CAD 里左右件各自建模/装配而非同一实体的镜像。

**已修复（commit `044959e`）**：左侧质量改为与右侧同值 0.333676，URDF /
`pikachu_v031.xml` / `pikachu_v031_train.xml` 三处同步；整机质量
6.759964 → 6.770275 kg（+0.153%）。复跑 `diag_v031_symmetry.py` 后第 1 节所有
`dm(g)` 归零，第 5 节整机镜像残差仍是 0.000000000 mm，改动没有破坏原有镜像
运动学。**残留未改**：该连杆质心镜像残差 0.580 mm、惯量相对差 0.51%
（`hip_pitch_link` 0.068 mm、`knee_up_link` 0.35%），属左右两件 CAD 零件的固有
差别，本次只统一了质量项。

### 那 0.58 mm 质心镜像残差是什么（`diag_v031_symmetry.py` 第 6 节）

先把一个可能的误解排掉：**不是 link frame 定义偏了**。第 5 节的 FK 残差恰好是
`0.000000000 mm`，说明左右 link frame 严格镜像；残差完全长在导出的 `inertial`
数据里。把两个 STL 当闭合实体（发散定理积分）算出来之后，0.58 mm 拆成两层：

| 量 | 数值 |
|---|---:|
| 实体体积 L / R | 299.193 / 300.460 cm³（**差 0.422%**）|
| 三角化面数 L / R | 85663 / 90733（不同）|
| **网格体积质心镜像残差** | **0.148 mm** |
| XML `inertial pos` 镜像残差 | **0.580 mm** |
| 网格质心 vs XML `inertial pos` 距离 | L 0.980 mm / R 1.171 mm |

- **0.148 mm 是真实几何差异**：两件体积差 0.42%、三角化面数不同，说明它们在 CAD
  里是各自建模/各自网格化的两个零件，不是"镜像同一个实体"。
- **放大到 0.580 mm 是导出误差**：XML 的 `inertial` 来自 CAD 在**精确 B-rep 实体**
  上算的质量属性，而 STL 只是三角化近似 —— 同一零件两者就差约 1 mm；左右两件的
  三角化误差方向/大小不同，这个差就叠成了额外的 0.43 mm。

顺带查了**质量与惯量是否自洽**（形状相同时 `I ∝ m·L² ∝ V^(5/3)`，故 `m ∝ I^(3/5)`）：
两件归一化惯量形状在 0.2% 内一致，所以惯量反推的质量比 `0.997990`、体积反推
`0.995785`，而修复前 XML 声明的质量比是 `0.969093` —— **左件的质量比它自己的惯量
低约 3.1%，与几何无关**，更像一次错误的 mass override（SolidWorks 的 Override
Mass Properties 可以只改质量、保留几何算出的惯量，正好产生这种不自洽）。
这也解释了为什么其他 Pikachu 变体（27dof / sl_v21 / sl_v22 / v03 / V025_lite）
的同一个连杆左右质量都是 0 差，只有 v031 有 10 g 的缺口。修复后声明值变成
`1.000000`，落回惯量反推值的 0.2% 以内 —— 即这次修复是把异常的那个数拉回几何
自洽的位置，不是凭空造数。

换算成 hip_yaw 关节处的重力力矩量级：该关节轴在 body frame 原点、方向
`[0, 0, 1]`，两侧 COM 到轴的垂直距离约 **2.43 cm**。注意**零位站立时该轴竖直，
重力绕它的力矩恰好为 0**；只有腿摆动使轴倾斜 θ 时才有 `Δm · g · d · sinθ`，
最大（θ = 90°）约 `0.010311 × 9.81 × 0.0243 ≈` **2.46 mN·m**，占 hip_yaw 电机
effort_limit（12.5 N·m）的 **0.02%**。这个量级不足以解释观测到的 0.6–1.0 rad
目标越界与 12 s 内约 100° 的偏航漂移 —— 它至多是"对称性破缺的种子"，不是原因
本身。修复它属于"把模型收拾干净"，预期不会改变行为；本轮用同种子重跑一档 T6
（见下）直接量这个差值。

### 尚不能完全排除的复现风险

2026-10-10 那次训练时，V031 的 constants、任务配置和训练 XML 都还没有被 git
跟踪，run 内的 `git/mjlab.diff` 因而没有保存这些文件，run 目录也没有资产快照。
`env.yaml` 保存了大部分有效参数，但没有保存完整 MJCF 惯量、mesh、等式约束和
碰撞几何。这些文件已在本次工作中入库（`4fb5e18` 起），但**对历史那次 run 的
缺口无法事后补上**。因此本报告能证明"**当前 XML** 没有明显 hip_roll 左右错误"，
但不能密码学地证明 2026-10-10 训练时读取的 XML 与当前文件逐字相同。

本次回放/诊断所用文件的 SHA-256（质量修复后）：

```text
d68be1ade235303ccc2230dedbfa286051834d1d6728e3453eddf4fd41507866  pikachu_v031_train.xml
710fe68e4145a04027f576ea13825bcd98010cba8982568839720f043b0f4802  pikachu_v031_constants.py
10d806adb8063523bc2b4a0ecedeb89da1141195c98f8776b40588fa74fa1625  pikachu_v031/env_cfgs.py
```

修复前的 `pikachu_v031_train.xml` 是
`4211929cf76cc8c261cb42a035e36f19186ece200333d8a71f2479d7ffb56ee7`，与上面的
差异只有 `left_hip_yaw_link` 一行质量。

## 改进方向与实验顺序

### P0：动作安全边界 → 已落地为 Clip / SymClip 档

1. 在 V031 runner 配置中设置 `clip_actions=1.0`，使 hip_roll / hip_yaw 的
   处理后目标限制在 ±0.20 rad，给 ±0.26 rad 机械限位留余量。
2. 在 `JointPositionActionCfg.clip` 再加一层逐关节处理后目标边界，防止以后
   修改全局 runner 时失去安全边界。
3. **从头训练**，不要在 m29999 上直接打开 clip 后继续部署；上面的消融已证明
   旧策略依赖饱和控制。
4. 新增 `raw_action_clip_fraction`、`processed_target_limit_fraction` 和每关节目标
   p99/max 指标。验收要求：处理后目标越界率严格为 0。

只做这一项会改变策略可用的控制权，回报可能先下降；这是预期现象，不应通过放开
限位恢复旧回报。

### P1：加入正确的左右镜像学习 → 已落地为 Sym / SymClip 档

优先使用 RSL-RL 已支持的 symmetry extension：

- `use_data_augmentation=True`；
- `use_mirror_loss=True`；
- `mirror_loss_coeff` 从 0.1、0.3、1.0 做小网格；
- 实现 V031 专用 mirror function，同时处理 actor、critic 和 action。

镜像不能只交换左右索引。以 sagittal reflection 为例：

- 线速度/重力等极向量：`(x, y, z) -> (x, -y, z)`；
- 角速度等轴向量：`(wx, wy, wz) -> (-wx, wy, -wz)`；
- 命令：`(vx, vy, wz) -> (vx, -vy, -wz)`；
- 左右关节交换时必须使用 V031 实测符号表。驱动关节中
  `hip_pitch/hip_roll/hip_yaw/ankle` 反号，`knee_up` 同号；
- 双脚高度、air time、contact 交换；contact force 还要做向量镜像。

不要直接加“同一时刻左右关节角相等”的奖励；双足交替步态包含半周期相移，朴素
L/R L2 会压制正常迈步。镜像数据增强或 policy mirror loss 更合适。

### P1：为大偏航误差补非饱和梯度 → 已落地为 YawHuber 档

仿照现有 `linear_velocity_huber_error` 增加
`angular_velocity_huber_error`，至少覆盖 `wz_cmd - wz_actual`，建议初始
`delta=0.25 rad/s`、权重在 `-0.2 ~ -0.5` 做消融。保留指数奖励负责零附近精度，
Huber 项负责大误差恢复。

同时单独记录 body roll/pitch rate 与 yaw rate，避免当前
`body_angular_velocity_penalty` “只罚 xy、不罚 z”造成误读。

### P2：分阶段加入扰动 → 未做，留第二轮

建议三阶段课程，而不是从第 0 轮每 1–3 秒注入满范围扰动：

1. 先在 nominal + 小观测噪声下学出直行和镜像步态；
2. 加入较弱推力，例如横向 ±0.15 m/s、yaw ±0.20 rad/s、间隔 4–8 秒；
3. 最后再逐步扩大到目标鲁棒范围。

当前 actor 的 base linear velocity 噪声 ±0.5 m/s 与命令范围同量级，也建议做
0.1 / 0.25 / 0.5 三档消融，避免训练初期观测被噪声主导。

### P2：固定评测协议 → 已落地为 `scripts/eval_v031_symmetry.py`

每个候选配置至少跑 5 个训练 seed；每个 checkpoint 对 20–100 个评测 seed 测试：

- stand、0.30/0.40 m/s 前进、后退、左右侧移、左右转向；
- nominal、startup DR、周期推力三类条件；
- 记录 20 秒成功率、每米 heading drift、yaw RMS、横向速度 RMS；
- 记录左右 hip_roll 的 DC 均值差、p2p 比值、目标越界率和扭矩饱和率。

建议第一轮验收线：

| 指标 | 门槛 |
|---|---:|
| 处理后 joint target 越硬限位 | 0% |
| 直行 8 秒 heading drift | < 10° |
| 零 yaw 命令下 yaw RMS | < 0.25 rad/s |
| `|mean(|qL|) - mean(|qR|)|` | < 0.03 rad |
| 左右 hip_roll p2p 比 | 0.8–1.25 |
| 20 秒 nominal 成功率 | > 99% |
| 20 秒扰动成功率 | > 95% |

## 实际执行的实验矩阵

第一轮不追求"最终方案"，而是**单因素隔离**：每档相对 T6 基线只改一个因素，
这样每个因素的独立贡献才可归因；累积式方案留到第二轮。

| 档 | 任务 ID 后缀 | action clip | symmetry 增强 | yaw Huber | 隔离的因素 |
|---|---|---|---|---|---|
| T6 | `-V031-T6` | 无 | 无 | 无 | 基线 |
| Clip | `-V031-Clip` | ±1.0 | 无 | 无 | 动作裁剪 |
| Sym | `-V031-Sym` | 无 | 有 | 无 | 左右镜像 |
| SymClip | `-V031-SymClip` | ±1.0 | 有 | 无 | 裁剪 × 镜像 |
| YawHuber | `-V031-YawHuber` | 无 | 无 | 有 | 偏航非饱和梯度 |

五档**同预算**：4096 env × 5000 iter，`save_interval=500`，`WANDB_MODE=disabled`。
一键脚本：`scripts/run_v031_symmetry_ablation.sh`（前四档串行）、
`scripts/run_v031_arm.sh <task_id>`（任意档，供追加）。

**资产修复与档次的时序**：T6 档在质量修复落盘前就已启动（进程启动时已把旧模型
读进显存），Sym / SymClip / Clip / YawHuber 四档加载的是修复后的
`pikachu_v031_train.xml`。为避免"基线与其他档不是同一份资产"这个方法论漏洞，
队列尾部追加了一次**修复后的 T6 重跑**（`scripts/run_v031_arm.sh` 同参数、
种子固定为 `cfg.seed=42`），落进同一个 `pikachu_v031_velocity/` 实验目录的新
时间戳子夹。因此这张表里 T6 的最终口径与其余四档同资产；修复前的那次 T6 保留
在原时间戳子夹里，作为"同种子、只差 10.3 g 资产"的**配对对照**，可直接量出这次
质量修复对指标的影响。差值预期为零（力矩量级 0.02% of effort_limit）。

> **`clip_actions` 是 runner cfg 的字段**，不是 algorithm 的。`train.py` 读
> `cfg.agent.clip_actions` 传给 `RslRlVecEnvWrapper`；写成
> `cfg.algorithm.clip_actions` 不报错但**静默失效**，整档退化成基线。
> `tests/test_pikachu_v031_symmetry.py` 有专门的配置回归测试捕获这一点 ——
> 把它改回错写法时，正好只有 Clip 和 SymClip 两档失败。

**当前进度（2026-10-12 22:0x，训练已暂停）**：T6 档 5000 iter 跑完并已评测
（见下一节）。Sym 档在 **808 / 5000 iter** 时按用户要求暂停，run 目录
`logs/rsl_rl/pikachu_v031_velocity_sym/2026-10-12_22-01-20/`；Clip / SymClip /
YawHuber 三档与队列尾部的"修复后 T6 重跑"都**尚未开始**。暂停时的 GPU 已确认
空闲。恢复方案见「新 T6 档评测结果与跨 run 可比性核查」之后的小结。

优先看 m1000 / m3000 / m5000 而不是等到 30000 轮：如果 processed target 已长期
贴 clip、heading drift 不降或 mirror loss 不收敛，应立即停训调整。

## 新 T6 档评测结果与跨 run 可比性核查

T6 跑完 5000 iter 后（`logs/rsl_rl/pikachu_v031_velocity/2026-10-12_20-05-34`），
按上面同一份协议（`--seconds 8 --seeds 42,7`，7 条命令）评测，并把旧基线
（`2026-10-10_20-11-53`）的 m1000 / m3000 也补齐，凑成两条完整曲线。

### 1. 两条曲线

fwd 命令（0.4 m/s，8 s）逐指标，seed 42 / 7：

| 档 | ckpt | Δyaw° | °/m | hr_d | p2p 比 | 越界率 |
|---|---:|---:|---:|---:|---:|---:|
| newT6 | 1000 | 47.8 / 45.8 | 15.4 / 14.8 | 0.0866 / 0.0868 | 0.680 / 0.683 | 0.910 / 0.907 |
| newT6 | 3000 | 25.4 / 34.4 | 8.0 / 10.8 | 0.0960 / 0.0962 | 0.556 / 0.557 | 0.873 / 0.875 |
| newT6 | 4999 | 25.1 / 26.8 | 8.0 / 8.6 | 0.0681 / 0.0680 | 0.763 / 0.763 | 0.915 / 0.915 |
| oldT6 | 1000 | 90.9 / 92.3 | 28.3 / 28.8 | 0.0648 / 0.0654 | 0.710 / 0.716 | 0.853 / 0.858 |
| oldT6 | 3000 | 21.0 / 21.5 | 6.9 / 7.0 | 0.0767 / 0.0770 | 0.707 / 0.701 | 0.880 / 0.882 |
| oldT6 | 5000 | 126.8 / 120.5 | 39.4 / 37.4 | 0.0759 / 0.0765 | 0.687 / 0.685 | 0.850 / 0.850 |
| oldT6 | 29999 | 59.4 / 92.4 | 18.2 / 27.7 | 0.0569 / 0.0569 | 0.836 / 0.864 | 0.880 / 0.900 |

全命令聚合与镜像等变：

| 档 | ckpt | agg °/m | agg yawRMS | \|hr_d\|max | 越界 max | 镜像等变 | 摔倒 | stand hr_d | stand yawRMS |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| newT6 | 1000 | 15.9 | 1.134 | 0.0868 | 0.990 | 0.729 | 否 | 0.0006 | 0.0458 |
| newT6 | 3000 | 6.1 | 1.138 | 0.0962 | 0.990 | 0.704 | 否 | 0.0023 | 0.0409 |
| newT6 | 4999 | 16.5 | 1.133 | 0.0681 | 0.990 | 0.708 | 否 | 0.0039 | 0.0469 |
| oldT6 | 1000 | 17.6 | 1.159 | 0.0654 | 0.993 | 0.714 | 否 | 0.0081 | 0.0469 |
| oldT6 | 3000 | 2.9 | 1.221 | 0.0770 | 0.990 | 0.754 | 否 | −0.0004 | 0.0715 |
| oldT6 | 5000 | 20.8 | 1.255 | 0.0765 | 0.993 | 0.755 | 否 | 0.0047 | 0.1160 |
| oldT6 | 29999 | 12.3 | 1.257 | 0.0569 | 0.993 | 0.781 | 否 | 0.0095 | 0.0608 |

newT6@4999 逐命令 Δyaw°（seed 42）：stand 2.2 / fwd 25.1 / back 86.5 /
left 143.0 / right 133.8 / turnL 1.8 / turnR 1.9。

### 2. 现象在新 T6 上稳定复现，且不限于 fwd

对 newT6@4999 逐条对验收线：越界率 0.915（要求 0）、fwd 8 s 偏航 25.1°/26.8°
（要求 <10°）、fwd `|hr_d|` 0.068（要求 <0.03）、fwd p2p 比 0.763
（要求 0.8–1.25）—— **四条不合格**。零 yaw 命令下 stand 的 yaw RMS 0.047
（要求 <0.25）合格。8 s 内无摔倒。

更重要的是**失败模式高度一致**：7 个 checkpoint 里，只要命令带平移
（fwd/back/left/right），Δyaw 就在 21–158° 之间，越界率 0.85–0.99；
而 stand / turnL / turnR 三条命令在**每一个** checkpoint 上都干净
（Δyaw ≤6.6°，stand hr_d ≤0.0095，stand 越界率 0.000）。也就是说：

- 这不是"旧 run 偶发的一条坏轨迹"，而是**策略在带平移的步态下系统性地
  选到了非镜像解**；站立与原地转向（不需要换步）没有这个问题。
- 越界率、`hr_d`、`mirror_equiv`（0.70–0.78）三个指标在 7 个 checkpoint、
  2 个 run 上全部落在同一窄带内，**是稳定的、可复现的失败签名**。

### 3. 但 Δyaw 这个指标本身噪声极大，单点对比不可用

旧 run 自己的 4 个 checkpoint：m1000 **90.9°** → m3000 **21.0°** →
m5000 **126.8°** → m29999 **59.4°**。同一次训练、同一个种子、同一份资产，
相邻 checkpoint 之间就能差 6 倍。newT6@4999 的 25.1° 落在旧 run 的
21.0–126.8° 区间**内部**，oldT6@3000 的 21.0° 甚至比它更好。

因此：**不能把"旧 T6@5000 漂 120°、新 T6@4999 漂 27°"读成新 run 变好了
4.5 倍**——那是拿两个噪声样本相减。单 seed、单 checkpoint 的直行偏航只能
分辨量级差异（比如"会不会倒"、"是不是 100° 以上"），分辨不了 2 倍以内的差别。
这条限制在第二轮必须处理：要么每档多 seed，要么以越界率 / `hr_d` /
`mirror_equiv` 这三个窄带指标为主判据。

### 4. 旧 run 不能当同口径参照，但原因不是代码/资产/配置

逐项核对两次 run，全部一致或有明确无害解释：

| 项 | 旧 run `2026-10-10_20-11-53` | 新 run `2026-10-12_20-05-34` | 结论 |
|---|---|---|---|
| 主机 / GPU | `finnox-4090` / RTX 4090 | `finnox-Legion-R7000-APH9` / RTX 4060 Laptop | **不同**（tfevents 文件名可见） |
| `pikachu_v031_train.xml` SHA-256 | `4211929c…` | `4211929c…` | 同（与报告时、与 `4fb5e18` 内均一致） |
| `params/env.yaml` md5 | `bd9a50d8a46e5cad04a107e3d115801e` | 同 | 逐字节相同 |
| `params/agent.yaml` | `max_iterations: 30000` | `5000` + `symmetry_cfg: null` | 只差循环上界（`train.py:198` 传给 `num_learning_iterations`，无调度作用）+ 一个默认 `None` 的新字段 |
| seed | 42 | 42 | 同 |
| 未提交代码 | 33 文件 / 3328 行，基线 `ee119be`（该提交已不在库中） | 4 文件 / 266 行，基线 `4fb5e18` | 见下 |
| `Train/mean_reward` @1000 / 3000 / 5000 | 49.36 / 47.05 / 49.95 | 50.01 / 45.65 / 48.33 | 同量级 |

未提交代码的核对方式：把旧 run 那份 `git/mjlab.diff` 里**每一段新增行**拿去
`4fb5e18` 里逐块查找，**81/90 块已经落地**（即旧 run 的工作区内容后来被提交成了
`4fb5e18`）。剩下 9 块分布在 `.gitignore`、`tests/test_velocity_*.py`、
`tasks/registry.py`（任务池展示顺序）和 `velocity/mdp/rewards.py` 的两个函数
（`short_step_air_time`、`feet_lateral_distance`）——后者是 v03 步态实验用的项，
`env.yaml` 逐字节相同已经证明 v031-T6 没有引用它们。**所以 v031-T6 这条路径上
两次 run 的代码是同一份。**

综合起来：两次 run 唯一的实质差异是**机器/GPU**。这既可能通过数值非确定性
把 PPO 推向不同解，也可能是本次没能排除的其它环境因素；但既然 Δyaw 在同一个
run 内部就有 6 倍波动（第 3 节），**不需要引入"代码不同"来解释这个差值**。
处置办法就是不再拿旧 run 做配对参照，改以本轮 newT6 为基线。

### 5. 小结：剩余四档该不该继续训

**该继续，但不要"接着排队往下跑"，要重开一批。** 理由与改动：

1. **资产口径要统一。** T6 是质量修复前启动的，Sym / SymClip / Clip / YawHuber
   是修复后启动的。10.3 g 的修复本身量级可忽略（0.02% of effort_limit），但
   既然要重开，就没有理由留这个方法论漏洞：五档应当在**同一份
   `pikachu_v031_train.xml`** 上跑，T6 作为批次内的一档重跑，而不是队列尾部的
   补丁。
2. **主判据换成窄带指标。** 第 3 节证明 Δyaw 在单 run 内部就能差 6 倍，所以
   五档对比的主判据应是 `target_violation`、`hr_d`、`mirror_equiv`、`p2p 比`
   这四个跨 checkpoint 稳定的量；`heading_drift` 只用来判断"有没有走成弧线"
   这种量级差异。这条改动**不需要多花 GPU 时间**，只影响怎么读结果。
3. **优先级建议：Clip 与 Sym 先跑，YawHuber 后置。** 越界率 0.85–0.99 是
   7 个 checkpoint 上唯一"全员一致且离验收线最远"的指标（要求 0%），而
   `clip_actions=±1.0` 是直接针对它的最小改动；`mirror_equiv` 0.70–0.78 是
   第二个稳定的失败签名，`Sym` 直接针对它。YawHuber 针对的 `track_angular_velocity`
   饱和在 stand / turnL / turnR 上并没有表现出问题（Δyaw ≤6.6°），先验上
   优先级最低。
4. **预算**：本机 4060 Laptop 上 4096 env × 5000 iter 约 2 小时，五档串行约
   10 小时。若要先验方向，可把每档砍到 2000 iter（约 4 小时跑完五档）做筛查，
   再对活下来的档补满 5000 iter —— 第 1 节的曲线显示 m1000→m3000 已能看出
   越界率与 `hr_d` 的量级，筛查用 2000 iter 是有信息量的。

## 推荐最小实验矩阵（第二轮，累积式）

在单因素结论确定后，按下列顺序做累积式验证：

| 实验 | action clip | symmetry | yaw Huber | push curriculum | 目的 |
|---|---|---|---|---|---|
| E0 | 无 | 无 | 无 | 原配置 | 复现基线 |
| E1 | ±1 | 无 | 无 | 原配置 | 验证饱和控制根因 |
| E2 | ±1 | augmentation + loss | 无 | 原配置 | 验证左右 DC 偏置 |
| E3 | ±1 | augmentation + loss | 有 | 原配置 | 验证偏航恢复 |
| E4 | ±1 | augmentation + loss | 有 | 分阶段 | 最终鲁棒方案 |

## 本轮实验的已知局限

在解读五档对比时必须带上这三条，否则容易高估结论强度：

1. **每档只有 1 个训练 seed。** 验收线里写的是"每个候选配置至少 5 个 seed"，
   但本机只有一块 8 GiB 的 GPU，五档串行已经是约 9 小时，5 seed 需要约 45 小时。
   因此本轮只能看**档间差异是否大到超出 seed 噪声**，不能对小幅差异下结论。
   真要判定"某档更好"，必须补 seed。
2. **只训 5000 iter，而基线跑了 30000 iter。** 这是刻意的同预算单因素对比。
   用旧基线的曲线校验过这个预算够不够：旧 T6 的 `Train/mean_reward` 在
   500 iter 为 43.36、1000 为 49.36、5000 为 **50.01**、30000 为 **53.82** ——
   5000 iter 已拿到 30000 iter 的约 **93%**，之后是缓慢爬升。所以 5000 iter 做
   **档间对比**是站得住的。但仍要注意 Sym / SymClip 的镜像增强让有效 batch
   翻倍，学习动态与基线不同，早停点未必在同一个 iter；若某档曲线到 5000 iter
   仍在明显上升，结论应记为"预算不足，待续训"，而不是"该因素无效"。
3. **评测 seed 只有 2 个（42, 7）。** 单条 rollout 的 yaw 漂移方差不小，跨 seed
   取均值只压掉一部分。`hr_d`（同向分量）这类跨 seed 全同号的系统量更可信，
   `heading_drift` 的绝对值则要谨慎。**2026-10-12 的补充评测把这条从"猜测"
   变成了实测**：旧 T6 同一 run 的 4 个 checkpoint，fwd Δyaw 依次是
   90.9 / 21.0 / 126.8 / 59.4°，同种子同资产相邻 checkpoint 就差 6 倍。
   所以本轮的 `heading_drift` 只作定性判据（"是否走成弧线"），定量比较必须
   落到越界率 / `hr_d` / `mirror_equiv` 这三个窄带指标上。
4. **两轮 T6 跑在不同机器上。** 旧 run 在 `finnox-4090`（RTX 4090），新 run 在
   `finnox-Legion-R7000-APH9`（RTX 4060 Laptop）。逐项核对后代码、资产、
   `env.yaml`、`agent.yaml`（只差 `max_iterations`，仅作循环上界）、seed 均相同，
   但 GPU 不同会通过数值非确定性把 PPO 推向不同解。因此本轮**以 newT6 为基线**，
   旧 run 只作为"现象在修复前的原始数据上也存在"的旁证，不做配对比较。
   若第二轮要跨机器比，应固定机器。

## 复现

重新生成曲线、训练标量摘要和异常遥测视频：

```bash
cd mjlab
uv run python doc/pikachu_v031_velocity_20261010/analyze_run.py
```

报告生成脚本只读取原日志和本报告目录中的回放 CSV，不修改 checkpoint 或训练日志。

对五档消融的 checkpoint 跑同口径评测并汇总成一张对比表（**训练结束后再跑**，
否则与训练抢 GPU）：

```bash
cd mjlab
MUJOCO_GL=egl uv run python scripts/eval_v031_all_arms.py \
    --checkpoints 1000,3000,4999 --seconds 8 --seeds 42,7 \
    --out /tmp/v031_eval
```

它复用 `scripts/eval_v031_symmetry.py` 的评测协议，不重写指标口径，产物为
`<out>/<档>/model_<ckpt>/eval.json` 与汇总 `<out>/summary.json`。

单项诊断（资产左右对称性，纯 CPU，可与训练并行）：

```bash
uv run python scripts/diag_v031_symmetry.py
```
