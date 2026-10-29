# Pikachu V031 无动作边界的实机部署风险调研报告

**对象**：Pikachu V031 velocity policy，重点检查
`pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42/model_49999.pt`<br>
**问题**：策略动作输出没有 clip 时，对实机部署有什么影响<br>
**日期**：2026-10-12<br>
**结论级别**：代码审计 + 官方资料调研 + 仿真遥测；不是实机放行测试

## 结论

**会有显著影响。当前 Mirror005 checkpoint 不应直接部署到实机。**

“没有 action clip”不等于电机会必然立刻损坏：如果实机控制链还有正确的关节目标
软限位、速度/加速度限制、电流/力矩限制、温度保护、通信 watchdog 和急停，后级
保护可以阻止部分最坏结果。但这些保护只能限制损害，不能保证策略仍能平衡或行走；
任何一级缺失、参数/符号错误或未覆盖 ONNX 直出路径，都可能造成跌倒、撞机械限位、
持续电流饱和或过热。

对当前模型，风险不是理论上的：

- 0.4 m/s nominal 直行时，**36.88% 的动作元素超过 ±1**；跳过 0.5 秒瞬态后，
  **100% 的控制周期至少有一个动作超过 ±1**。
- raw action 最大绝对值为 **3.32**；侧移测试最高为 **4.23**。
- 直行中，10 个关节目标合计有 **18.08%** 超出各自机械位置范围，且每个控制周期
  至少有一个目标越机械范围。
- 左右 hip-roll 目标越 ±0.26 rad 的比例为 38.67% / 40.80%；hip-yaw 最大目标约
  ±0.664 rad，是机械范围绝对值的约 2.55 倍。
- 目标指令相邻 20 ms 周期的最大变化折算为 **28.13 rad/s**；快速前进达到
  **36.37 rad/s**。这不是实际关节速度，但表示低层位置环会收到很陡的设定值跳变。
- 左右踝的仿真扭矩都已经碰到 **3.0 N·m** 限幅；直行稳态中分别有 14.13% /
  12.27% 的样本超过限幅的 80%。
- 导出的 ONNX 图中只有 `Gemm/ELU/Sub/Div`，**没有 `Clip`、`Tanh` 或
  `Sigmoid` 节点**。runner 和环境里的限幅不会自动进入当前纯策略 ONNX。
- 仓库中没有找到可审计的实机 CAN/电机 runtime、ONNX 后处理、watchdog 或急停
  实现，因此目前没有证据证明部署端存在完整的最后安全屏障。

最终判定：**当前问题不是“要不要在实机端随手加一行 clip”，而是训练、导出和
实机控制链必须使用同一个、有版本记录的有界动作协议。**

## 先区分五种容易混淆的“clip”

| 层级 | 本质 | 能防止什么 | 不能替代什么 |
|---|---|---|---|
| PPO `clip_param=0.2` | 限制 PPO probability ratio/value update | 过大的策略更新 | **不是动作限幅** |
| raw action bound | 把策略动作限制到如 `[-1,1]` | 无界网络输出 | 不能限制目标变化率、扭矩和温度 |
| processed target bound | 对 `offset + scale × action` 后的关节目标限幅 | 目标越安全角度范围 | 不能限制瞬时跳变 |
| slew/rate bound | 限制目标速度、加速度/jerk | 指令台阶和冲击 | 不能代替电流/温度保护 |
| effort/current + hard limit | 电机和机械最后保护 | 峰值扭矩、硬行程 | 不能保证机器人不跌倒或策略正确 |

RSL-RL 文档把 `clip_param` 定义为 PPO surrogate/value clipping，同时明确普通
`GaussianDistribution` 的 mean 由 MLP 直接产生；它还提供天然有界的
`BetaDistribution`。两者是完全不同的概念。
[RSL-RL 官方配置文档](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst)

## 当前 V031 的真实动作链

### 代码审计

| 环节 | 当前 Mirror005 配置/行为 | 结论 |
|---|---|---|
| actor distribution | `GaussianDistribution`，`init_std=1.0` | 支持集无界 |
| 训练末期 mean std | 0.268 | 探索方差不大，但 mean 仍无界 |
| runner | `clip_actions=None` | 进入环境前不裁剪 |
| action term | `JointPositionActionCfg.clip=None` | scale 后目标不裁剪 |
| hip-roll/yaw scale | 0.20 rad / action unit | `|a|=3.3` 可产生约 0.66 rad 目标 |
| hip-roll/yaw hard range | ±0.26 rad | 不是策略目标范围 |
| hip effort limit | 12.5 N·m | 只约束仿真输出力矩 |
| ankle effort limit | 3.0 N·m | 直行中已触发饱和 |
| ONNX deterministic output | Gaussian mean 原样输出 | 不包含 runner/env 后处理 |

当前默认关节 offset 为零，因此：

```text
q_des = action_scale × raw_action
```

例如 hip-roll/yaw 的 `raw_action=1` 对应 ±0.20 rad，距离 ±0.26 rad 机械限位还有
0.06 rad 余量；但 `raw_action=3.3` 对应约 0.66 rad。最后线性层没有边界时，观测
噪声、状态分布偏移或异常输入还可能产生比已记录值更大的输出。

Isaac Lab 的 action term 同样把 action 定义为
`offset + scale × input_action`，并把 `clip` 作为独立配置项；这与本项目实现一致。
[Isaac Lab 官方 action API](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.envs.mdp.html)

### 导出的 ONNX 不携带安全后处理

对当前
`2026-10-12_00-11-23_mirror005-s42.onnx` 的图节点检查结果：

```text
Gemm: 4
Elu: 3
Sub: 1
Div: 1
Clip/Tanh/Sigmoid: 0
```

`Sub/Div` 来自 observation normalization；输出端是最后一个线性层。即使以后只在
`RslRlVecEnvWrapper` 设置 `clip_actions=1.0`，当前 exporter 仍然只导出 policy，
wrapper clip 和 `JointPositionActionCfg.clip` 都不在 ONNX 内。实机 runtime 必须
显式执行相同后处理，或者改用导出时自身有界的分布/包装模块。

## 当前模型的定量风险

### 0.4 m/s 直行：逐关节风险

![逐关节风险](plots/joint_risk_summary.png)

| 关节 | raw max | 目标越硬范围 | 目标误差 RMS | 目标变化率 p99 | 扭矩 >80% 限幅 |
|---|---:|---:|---:|---:|---:|
| left hip-roll | 2.82 | 38.67% | 0.241 rad | 7.95 rad/s | 0% |
| right hip-roll | 2.87 | 40.80% | 0.246 rad | 7.79 rad/s | 0% |
| left hip-yaw | 3.31 | 33.07% | 0.248 rad | 9.01 rad/s | 0% |
| right hip-yaw | 3.32 | 30.93% | 0.248 rad | 8.88 rad/s | 0% |
| left knee-up | 2.33 | 19.20% | 0.407 rad | 21.89 rad/s | 0% |
| right knee-up | 2.37 | 18.13% | 0.405 rad | 22.70 rad/s | 0% |
| left ankle | 1.84 | 0% | 0.193 rad | 16.38 rad/s | **14.13%** |
| right ankle | 1.36 | 0% | 0.181 rad | 16.37 rad/s | **12.27%** |

hip 关节没有打到 12.5 N·m，不能被解释为“目标是安全的”：当前仿真 PD 增益较低，
且关节限位/执行器限幅吸收了错误。实机内置位置环的增益、采样率、摩擦、总线延迟和
电流限制如果不同，相同的目标误差会产生不同的峰值电流和力矩。

![目标与力矩时序](plots/fwd_target_and_effort_trace.png)

### 六类命令

| nominal 命令 | raw max | 动作元素 `|a|>1` | 有任一越界动作的周期 | 目标越机械范围 | 任一目标越界的周期 | 任一扭矩 >80% 的周期 |
|---|---:|---:|---:|---:|---:|---:|
| stand | 0.88 | 0% | 0% | 0% | 0% | 0% |
| fwd 0.4 | 3.32 | 36.88% | **100%** | 18.08% | **100%** | 26.40% |
| fwd 0.5 | 3.61 | 41.63% | **100%** | 22.00% | **100%** | 18.93% |
| back −0.3 | 3.47 | 34.03% | **100%** | 15.36% | **100%** | **65.33%** |
| lateral 0.25 | **4.23** | 28.37% | **100%** | 16.19% | 94.93% | 25.33% |
| turn 0.3 | 0.96 | 0% | 0% | 0% | 0% | 0% |

静止和当前失败的转向策略看起来“安全”，平移时才全面进入无界/饱和区。因此只做
站立台架测试不足以放行步态。

机器可读数据：

- [逐关节风险 CSV](data/joint_risk_summary.csv)
- [六命令风险 CSV](data/command_suite_risk.csv)
- [完整 JSON 摘要](data/summary.json)

## 对实机的具体影响

### 1. 更容易持续顶机械限位

位置目标远超硬范围时，位置环会持续尝试缩小不可能消除的误差。仿真里的关节限位是
约束求解器的一部分，并非“把目标自动改回安全值”；MuJoCo 官方文档也明确区分
joint range、control range 和 force range。
[MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)、
[MuJoCo limit computation](https://mujoco.readthedocs.io/en/latest/computation/index.html#limit)

实机后果取决于固件和结构，可能表现为：

- 电机持续输出限流力矩，关节停在限位上；
- 减速器、轴承、连杆或四杆闭合约束承受额外载荷；
- 编码器零偏或机械装配误差让左右侧在不同位置先撞限位；
- 保护触发、电机掉使能，随后机器人跌倒。

### 2. 电流饱和、发热和跟踪失真

扭矩/电流限幅可以防止无限力矩，却会让不同的巨大目标都映射成同一个饱和输出。
策略无法再获得它期望的关节响应，闭环状态进入训练中未覆盖的区域。当前踝关节已在
仿真中触发 3 N·m 上限，说明真实驱动器的连续电流、峰值电流和温度模型必须单独验证。

### 3. 指令跳变和冲击

幅值 clip 只限制位置目标范围，不限制 `q_des[k]-q_des[k-1]`。当前 50 Hz policy
的目标变化率最高达 28–36 rad/s；若实机 KHz 位置环直接把每个 policy step 当新
setpoint，会形成台阶输入。需要单独的速度/加速度限制或参考轨迹整形。

### 4. 仿真—实机行为不一致

若训练时无 clip，实机端却有 clip，策略在实机看到的是不同的闭环动力学；若训练和
实机都 clip，但 PPO 仍按未裁剪 Gaussian 计算 log-prob/entropy，大量尾部动作会
映射到同一个边界动作，产生优化退化。已有研究把这视为有界连续动作的专门问题。
[Clipped Action Policy Gradient, ICML 2018](https://proceedings.mlr.press/v80/fujita18a.html)

本项目的 E1 正好展示了这种风险：从头使用硬 clip 后目标越界降至 0%，但
`Gaussian std + entropy=0.01` 形成饱和漏洞，m1000 的 68.85% 动作被裁剪，
确定性策略只会站立。结论不是“clip 不该加”，而是**动作分布、训练时执行边界和
部署后处理必须一致**。

### 5. 异常观测会被无界最后一层放大

确定性部署使用 Gaussian mean，不会采样探索噪声，但 mean 本身仍是无界线性层输出。
传感器尖峰、错误单位、关节顺序/符号错误、归一化统计不匹配、NaN/Inf 或超出训练
分布的姿态都可能生成更大的动作。clip 不能修复这些根因，但能提供必要的幅值防线；
runtime 还必须在推理前后做 finite/range/staleness 检查。

## 推荐的分层安全结构

![部署安全层](plots/deployment_safety_layers.png)

任何一层都不能删除后面的保护：

1. **有界策略分布**：训练时优先使用 RSL-RL 5.4.2 已提供的
   `BetaDistribution(action_range=(-1,1))`。Beta sample 和 deterministic mean
   都天然在范围内，并包含正确的概率密度变换。RSL-RL 官方文档明确把 Beta 用于
   bounded action spaces。相关研究也指出，物理约束下标准 Gaussian 会引入裁剪
   偏差。[Beta policy, ICML 2017](https://proceedings.mlr.press/v70/chou17a.html)
2. **runner 冗余保护**：仍设置 `clip_actions=1.0`，作为实现错误时的 guard。
3. **processed target 软范围**：保留当前已实现但 Mirror005 未启用的
   `_ACTION_TARGET_CLIP`：
   hip pitch ±0.50、hip roll/yaw ±0.20、knee-up ±0.40、ankle ±0.35 rad。
4. **目标变化率限制**：根据实机阶跃/正弦辨识结果设置每关节
   `v_cmd_max/a_cmd_max`；不要直接用 URDF 峰值速度当正常步态目标变化率。
5. **低层执行器保护**：驱动器电流/峰值力矩、连续力矩、温度降额、欠压/过压和
   编码器故障保护。
6. **系统保护**：观测时间戳、CAN 丢包 watchdog、policy deadline、跌倒检测、
   安全姿态/卸力状态和物理急停。

ROS 2 Control 也把 position、velocity、acceleration、jerk 和 effort 限制作为可独立
启用的 command limiter；这支持“幅值、变化率和力矩要分层处理”的设计。
[ros2_control Joint Limiting](https://control.ros.org/rolling/doc/ros2_control/hardware_interface/doc/joint_limiting.html)

## 推荐训练方案

### 首选：天然有界分布

```python
actor.distribution_cfg = {
  "class_name": "BetaDistribution",
  "action_range": (-1.0, 1.0),
}
runner.clip_actions = 1.0
joint_position_action.clip = {
  r".*_hip_pitch_joint": (-0.50, 0.50),
  r".*_hip_roll_joint": (-0.20, 0.20),
  r".*_hip_yaw_joint": (-0.20, 0.20),
  r".*_knee_up_joint": (-0.40, 0.40),
  r".*_ankle_joint": (-0.35, 0.35),
}
```

当前 symmetry callback 处理的是 10 维已执行/确定性动作，Beta 的外部动作仍为 10
维，因此从代码结构看可以兼容；但必须新增单元测试，验证：

- Beta actor 的 action shape、log-prob 和 ONNX 导出；
- 左右镜像仍为 involution；
- mirror loss 作用于 Beta deterministic mean；
- 导出 ONNX 对随机观测的输出始终在 `[-1,1]`。

### 备选：受约束 Gaussian

如果 Beta 训练不稳定，可保留 Gaussian，但要从训练开始同时使用 runner/target
clip，并限制 std、降低 entropy，再验证 deterministic actor。它是过渡方案；大量
动作贴 clip 时仍存在分布与执行动作不一致的问题，不能只看训练 reward。

### 最小实验矩阵

| 实验 | 分布 | runner clip | target clip | slew model | 目的 |
|---|---|---:|---:|---:|---|
| U0 | 当前 Gaussian | 无 | 无 | 无 | 只作历史基线，禁止上机 |
| B1 | Beta [-1,1] | ±1 | 有 | 无 | 验证有界策略是否能走 |
| B2 | Beta [-1,1] | ±1 | 有 | 训练/随机化延迟 | 缩小执行器 sim-to-real gap |
| B3 | 同 B2 | ±1 | 有 | 加实测 slew/actuator model | 候选实机策略 |

先 seed 42 做 m500/m1000 筛选，通过后再跑 42/43/44。不要把旧 Mirror005 权重直接
加 clip 后上机：当前每个平移周期都会改变至少一个动作；原 E0 权重临时加 ±1 clip
后 2.8 秒摔倒，已经证明 post-hoc clip 可以彻底改变策略行为。

## 实机 runtime 的最低后处理

```python
raw = onnx_policy(normalized_observation)
if observation_is_stale or not all_finite(raw):
  enter_safe_stop()

action = clip(raw, -1.0, 1.0)            # defensive even for bounded policy
q_des = default_q + action_scale * action
q_des = clip(q_des, q_soft_min, q_soft_max)
q_des = limit_velocity_accel(q_des, previous_q_des, dt)

if command_deadline_missed or not state_is_sane:
  enter_safe_stop()
send_position_target(q_des, current_and_temperature_limits)
```

必须把以下内容写入模型 manifest，而不是依赖部署人员记忆：

- observation 50 维的顺序、单位、归一化 mean/std；
- action 10 维的关节顺序、左右符号、scale、offset；
- raw/processed/rate/current limits；
- policy 频率、时间戳和 watchdog timeout；
- ONNX SHA-256、后处理配置 SHA-256、机器人 URDF/MJCF 版本。

## 上机前放行门槛

### 离线/仿真

- 所有 deterministic 命令档：processed target 越软范围 **0%**；
- raw action 超 `[-1,1]`：有界分布应为 **0%**；
- ONNX 随机测试、回放测试和 PyTorch 输出一致；
- 20 秒 nominal 与扰动回放均通过速度、yaw、跌倒门槛；
- target velocity/acceleration 不超过实测安全 envelope；
- 正负转向均有正确方向响应，不允许用站立伪装稳定。

### 台架

1. 机器人悬吊、脚不接地，电流/力矩上限先降至额定连续值的一小部分；
2. 只跑 standing，逐关节核对顺序、正负号、零位和 soft limit；
3. 注入 NaN、冻结观测、延迟、CAN 丢包，确认 watchdog 进入安全状态；
4. 低速单腿/双腿周期动作，核对目标、实际位置、电流和温度；
5. 有保护架的地面站立，再逐步开放低速行走；
6. 每阶段出现持续饱和、撞限位、温升过快或 watchdog 误行为即停止。

## 额外发现：电机规格必须先统一

当前有效 URDF 和 `pikachu_v031_constants.py` 使用：

- hip pitch/roll/yaw、knee-up：12.5 N·m，12.57 rad/s；
- ankle：3.0 N·m，15.71 rad/s。

但源 `pikachu_v031.xml` 的注释仍写着 hip roll/yaw/ankle 使用 DM-J3507、±9 N·m，
同一文件的实际 motor 元素却是 hip roll/yaw ±12.5、ankle ±3。仓库其他旧文档还存在
不同的型号/力矩组合。

这意味着当前仿真配置内部是自洽的，但**实机 BOM、减速比、连续/峰值力矩和热保护
尚不能从仓库文档唯一确定**。在这个问题解决前，不能按 12.5/3.0 N·m 直接配置
实机电流上限，也不能把本报告的仿真 torque ratio 当成实机热裕量。

## 限制

- 本报告没有访问真实电机固件、CAN runtime、电池/驱动器参数和温度记录。
- 当前定量数据来自 seed 42、CPU deterministic 回放；它证明目标越界和饱和依赖，
  但不是实机故障概率估计。
- “目标变化率”是相邻 policy setpoint 的斜率，不等于关节实际速度；它用于判断
  低层控制器输入有多陡。
- 实机安全边界最终必须由机械余量、电机 datasheet、驱动器设置和台架辨识共同确定。

## 复现

```bash
cd mjlab
uv run python \
  doc/pikachu_v031_action_clip_deployment_20261012/analyze_action_risk.py
```

脚本只读取 Mirror005 的固定命令遥测，生成本目录的 CSV、JSON 和 PNG，不修改训练
checkpoint。

## 资料来源

- [RSL-RL Configuration：Gaussian 与 bounded Beta distribution](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst)
- [Isaac Lab Action API：scale、offset 与独立 clip](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.envs.mdp.html)
- [MuJoCo XML Reference：joint/control/force limits](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [MuJoCo Computation：limit 是约束求解的一部分](https://mujoco.readthedocs.io/en/latest/computation/index.html#limit)
- [ros2_control Joint Limiting](https://control.ros.org/rolling/doc/ros2_control/hardware_interface/doc/joint_limiting.html)
- [Fujita & Maeda, Clipped Action Policy Gradient, ICML 2018](https://proceedings.mlr.press/v80/fujita18a.html)
- [Chou et al., Beta Distribution for Continuous Control, ICML 2017](https://proceedings.mlr.press/v70/chou17a.html)
