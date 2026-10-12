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
基线只改一个因素，矩阵与验收线见下文。结果待训练完成后补入本报告。

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

优先看 m1000 / m3000 / m5000 而不是等到 30000 轮：如果 processed target 已长期
贴 clip、heading drift 不降或 mirror loss 不收敛，应立即停训调整。

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
   （顺带：新旧两个 T6 是同配置不同 seed，1000 iter 处 49.36 vs 50.01，可作为
   seed 噪声约 ±0.7 的粗略参考 —— 只有两个样本，仅供量级判断。）
3. **评测 seed 只有 2 个（42, 7）。** 单条 rollout 的 yaw 漂移方差不小，跨 seed
   取均值只压掉一部分。`hr_d`（同向分量）这类跨 seed 全同号的系统量更可信，
   `heading_drift` 的绝对值则要谨慎。

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
