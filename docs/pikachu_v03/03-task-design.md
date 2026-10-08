# 03 · velocity 任务设计

**任务 ID**：`Mjlab-Velocity-Flat-Pikachu-V03` / `Mjlab-Velocity-Rough-Pikachu-V03`
**代码**：`src/mjlab/tasks/velocity/config/pikachu_v03/`
（`env_cfgs.py` / `rl_cfg.py` / `__init__.py`）

任务骨架复用 `make_velocity_env_cfg()`，下面只记 v03 的**偏离项**。

---

## 1 · 动作空间

10 个被驱动关节（无手臂），全部 `JointPositionActionCfg` + `use_default_offset=True`。

**动作缩放按 v03 真实关节限位逐关节定标**（不是照抄 14dof 的 0.5）：

```python
joint_pos_action.scale = {
  r".*_hip_pitch_joint": 0.50,   # 限位 ±2.45，行程最大
  r".*_knee_up_joint":   0.40,   # 限位 [-1.22,+0.52] 右 / 镜像左，全行程 1.74
  r".*_hip_roll_joint":  0.20,   # 限位 ±0.26  —— 极小！
  r".*_hip_yaw_joint":   0.20,   # 限位 ±0.26  —— 极小！
  r".*_ankle_joint":     0.35,
}
```

> `hip_roll` / `hip_yaw` 的限位只有 **±0.26 rad**（约 ±15°），比 14dof 小得多。
> 如果沿用 0.5 的缩放，策略输出 ±1 就会撞限位，`dof_pos_limits` 惩罚常年为负，
> 学出来的是「贴限位走路」的怪姿态。

**`knee_joint` / `knee_mid_joint` 绝对不能加 actuator**。它们是四杆机构的被动
关节，由 `<connect>` 约束决定；加位置源会让两个执行器抢同一个机构，破坏约束。

---

## 2 · 电机与遥测

| 分组 | 关节 | 电机 | 力矩上限 |
|---|---|---|---|
| J4310 | `hip_pitch`, `knee_up` | DM-J4310P-2EC-V1.1 | ±12.5 N·m |
| J3507 | `hip_roll`, `hip_yaw`, `ankle` | DM-J3507-2EC-3R | ±9 N·m |

`add_pikachu_telemetry_metrics(cfg, joint_names=PIKACHU_V03_JOINT_NAMES,
actuator_groups=PIKACHU_V03_ACTUATOR_GROUPS)` —— `pikachu_metrics.py` 已参数化，
14dof 任务走默认表不受影响。

`joint_names` 用 `PIKACHU_V03_JOINT_NAMES`（10 个），**不含被动的
`knee` / `knee_mid`** —— 它们的「目标值」由四杆约束决定，不由策略决定，
进超规格统计没有意义。

---

## 3 · 命令空间

```python
# r04 现值（最终）。r01/r02 是 0.25/0.25/0.25 + heading 0.3 + standing 0.1；
# r03 短暂试过 0.20/0.15/0.15（已回退，见下方 ⚠️）。
rel_forward_envs  = 0.25
rel_backward_envs = 0.25     # ← 显式给后退样本
rel_lateral_envs  = 0.25     # ← 显式给侧移样本
rel_heading_envs  = 1.0      # ← 真正控制「有多少环境练转向」的旋钮
rel_standing_envs = 0.05
ranges.lin_vel_x = (-0.4, 0.5)
ranges.lin_vel_y = (-0.35, 0.35)
ranges.ang_vel_z = (-0.3, 0.3)
ranges.heading   = (-3.14, 3.14)   # heading_command=True
```

> ⚠️ **这三个方向桶的占比直接决定「有多少环境练过转向」**。
> `_resample_command` 对 forward / backward / lateral 三种环境**都把
> `ang_vel_z` 强制置 0** —— 落进桶里的环境永远见不到 yaw 命令。
>
> 但「见过 yaw 的环境占比」**不是** `1 − 桶总量`，要连乘三件事：
>
> ```python
> P(见过 yaw) = [ rel_heading_envs × (1 − 桶总量) ] × (1 − rel_standing_envs)
> #             ↑ is_heading_env &= ~directional    ↑ standing 最后清零整个命令
> ```
>
> | 轮次 | 桶总量 | heading | standing | 见过 yaw 的环境 |
> |---|---|---|---|---|
> | r01 / r02 | 0.75 | 0.3 | 0.1 | **6.79%**（139 / 2048） |
> | r03 | 0.50 | 0.3 | 0.1 | **13.49%**（276 / 2048） |
> | **r04** | **0.75** | **1.0** | **0.05** | **23.75%**（486 / 2048） |
>
> **6.79% 意味着 `track_angular_velocity`（权重 2.0）在平均奖励里的最大贡献
> 只有 `2.0 × 0.068 = 0.14`，而 `track_linear_velocity` 是全部环境贡献的
> 2.0 —— 信号弱 15 倍。** 策略把 yaw 整个忽略掉是理性的。
>
> ⚠️ **r03 走错了旋钮**：它靠砍方向桶（0.75→0.50）换暴露面，事后被 r04 证伪
> —— 方向桶恢复后 back/lat 的退化（1.02→0.81 / 0.41→0.26）**一点没回来**。
> 真正的取舍在**线速度与角速度之间**：r04 的 `track_ang` 0.295→0.827
> 而 `err_xy` 0.472→0.634（见 [04-worklog.md](04-worklog.md)）。
> **该动的是 `rel_heading_envs`，不是方向桶。**
>
> 前进/后退在 r02 已经 100% 跟踪，不需要更宽的覆盖。

> ⚠️ **`heading_command=True` 让「转向」变成一个位置任务**，这一点很关键：
> `_update_command` 里 `ang_vel_z = clip(0.5 × heading_error, ±0.3)` ——
> 策略学的是「**转到目标朝向就停**」，**从来没见过「一直转」这种命令**。
> 拿恒定角速度去评测是分布外输入，会得到假的低分
> （见报告 §5.8 与 `scripts/eval_v03_turn.py`）。
>
> 另一个后果：heading 环境**同时抽随机线速度**，所以训练分布里
> **没有「纯转向」环境** —— 实测 r04 边走边转 90° 收敛 0.98，
> 原地转只有 0.18。想要原地转得**加一个 `lin=0, ang≠0` 的桶**（r05 候选）。

课程（`command_vel`）：

```python
[
  {"step": 0,         "lin_vel_x": (-0.4, 0.5), "lin_vel_y": (-0.35, 0.35), "ang_vel_z": (-0.3, 0.3)},
  {"step": 2000 * 24, "lin_vel_x": (-0.6, 0.8), "lin_vel_y": (-0.5, 0.5),   "ang_vel_z": (-0.5, 0.5)},
]
```

`2000 * 24` = 2000 次迭代 × `num_steps_per_env`。

> **命令必须覆盖后退与侧移**。只给前进样本时，策略会把「后退」当成前进的
> 反向映射去学，实际速度符号是错的 —— 这在 AMP 侧已经被实测踩过
> （见记忆 `amp-v4-behavior-direction`）。这里从一开始就把三个方向都放进
> 命令分布。

---

## 4 · 奖励

沿用 `make_velocity_env_cfg()` 的骨架，v03 覆盖：

| 奖励 | 权重 | v03 参数 | 理由 |
|---|---|---|---|
| `track_linear_velocity` | 2.0 | **std = 0.20**（骨架默认 √0.25 = 0.5） | 见下 ⚠️ |
| `track_angular_velocity` | 2.0 | **std = 0.40**（骨架默认 √0.5 ≈ 0.707） | 见下 ⚠️ |
| `pose` | 1.0 | 见下 | **只覆盖 10 个驱动关节** |
| `upright` | 1.0 | `base_link` | — |
| `body_ang_vel` | **-0.05** | `base_link` | 抑制机身摇晃 |
| `angular_momentum` | **-0.02** | `root_angmom` | 抑制甩腿带来的角动量 |
| `base_vertical_velocity` | **-0.5** | — | 抑制上下弹跳 |
| `base_height` | **-0.2** | target 0.25 / std 0.02 | 防蹲防跳 |
| `air_time` | **+1.0** | `feet_air_time`，`threshold_min 0.05` / `max 0.5` | **每步给分** → 脱离站桩鞍点 |
| `alternating_air_time` | **+3.0** | `alternating_feet_air_time`，target 0.22 s / std 0.08 | **只在落地给分** → 收成真交替步 |
| `flight` | **-2.0** | grace 0.04 s | 禁止双脚同时离地（不是跑步任务） |
| `foot_clearance` | **-4.0** | target 0.04 m | 抬脚别太高也别拖地 |
| `foot_swing_height` | **-2.0** | target 0.04 m | 同上 |
| **`feet_lateral_distance`** | **-0.25** | target 0.20 m / std 0.05 | **r05 新增**：站姿横向开度，见下 |
| `soft_landing` | **-2e-3** | — | 落地缓冲 |
| `action_rate_l2` | **-0.15** | — | 动作平滑 |
| `action_acc_l2` | **-0.02** | — | 动作二阶平滑（新增项） |
| `dof_pos_limits` | -1.0 | — | 骨架默认 |
| `self_collisions` | **-1.0** | force_threshold 10 N | 新增项 |

### ⚠️ track 奖励的 std 必须随命令范围缩放

`make_velocity_env_cfg()` 的默认 std（lin `√0.25 = 0.5`、ang `√0.5 ≈ 0.707`）
是配 G1 那种 ±3 m/s 大命令的。v03 的命令收窄到 lin ±0.5 / ang ±0.3 之后，
**「站着不动」的误差就只有 `mean|cmd| ≈ 0.29 m/s`**：

```
exp(-0.29² / 0.5²) = 0.72     →  站桩白拿 72% 的跟踪分
```

梯度压力随之消失，策略直接落向「站着不动」这个鞍点 —— **r01 就是这么失败的**
（`error_vel_xy` 平在 0.68~0.73 不降、`air_time ≈ 0`、`episode_length` 却顶满 1000）。
公式的精确验证：站桩理论值 `2·exp(-0.29²/0.25) = 1.43`，
而 r01 实测 `Episode_Reward/track_linear_velocity` = **1.392 ~ 1.453**，分毫不差。

这不是新坑 —— AMP 侧踩过一模一样的（记忆 `amp-track-std-must-match-command`）。

**lin 收到 0.20**（站桩只剩 0.13 分）。**ang 收到 0.40 而不是更窄**，因为：

- r02 实测 yaw 误差落在 **0.2~0.6 rad/s** 区间（行走时未受命令的偏航 0.2~0.33、
  原地转向命令 0.30 则完全没跟）。
- std=0.25 时该区间 `exp(-err²/0.0625)` 已经很小、梯度趋零：
  err=0.5 处梯度 −0.29；std=0.40 时是 **−1.31**。
- 放宽的代价是站桩白拿的角速度分变多（`exp(-0.15²/0.16) = 0.87`），
  但站桩已经被 lin std=0.20 和 air_time 摁死了，不靠这一项。

### `pose` 的 std 表

```python
asset_cfg.joint_names = PIKACHU_V03_POSE_JOINT_NAMES   # 只 10 个驱动关节
std_standing = {".*": 0.05}
# r03 现值。r01/r02 曾是 {hip_pitch 0.4, hip_roll 0.15, hip_yaw 0.15, knee_up 0.4, ankle 0.35}
std_walking  = {hip_pitch 0.5, hip_roll 0.25, hip_yaw 0.30, knee_up 0.5, ankle 0.45}
std_running  = {hip_pitch 0.5, hip_roll 0.2,  hip_yaw 0.2,  knee_up 0.5, ankle 0.35}
running_threshold = 2.0
```

> ⚠️ **`hip_roll` / `hip_yaw` 不能用 0.15 收着**。v03 这两个关节的机械限位
> 只有 **±0.26 rad**，而**转向必须动 `hip_yaw`、侧移必须动 `hip_roll`** ——
> std 给 0.15 等于「一动就罚」。r02 实测：原地转向命令 0.30 → 实际 0.005，
> 侧移只有 41%。r03 放到 0.30 / 0.25（接近限位量级）才算「不惩罚」。

> `variable_posture` 要求 std 字典**覆盖 `asset_cfg.joint_names` 里的每一个
> 关节**，漏一个就 KeyError。
>
> `knee_up` 的 std 取 0.4（比 14dof 串联膝小）：v03 的 `knee_up` 全行程只有
> 1.74 rad（右 `[-1.22,+0.52]`），比串联膝小得多，用大 std 等于放任它乱摆。
>
> `std_standing = 0.05` 很紧。注意零动作时 PD 柔度会让 `knee_up` 从 -0.200
> 被压到 -0.2455（误差 0.0455），此时 `pose` 项 ≈ exp(-0.83) = 0.44 ——
> 即**站着不动也只能拿 44% 的站姿分**。这是设计上的取舍：它给策略一个
> 「主动把腿撑直一点」的梯度，而不是让它瘫着。

### `base_height` 的 target

v03 微屈膝站姿 `base_link` 高 0.26728 m，`target_height` 取 **0.25**（比站姿低
约 1.7 cm），`std = 0.02`。留出约 1 个 std 的余量，让正常步态中的起伏不挨罚。

### ⚠️ `feet_lateral_distance`：为什么必须新加一项，而不是收紧 `pose`

**症状（r04 @29999 实测）**：站立时双脚只张开 **9.2 cm**，而默认 keyframe 是
**20.0 cm**。两个 `hip_roll` 都**顶在硬限位上**（left +0.254 / right -0.261，
限位 ±0.26）—— 策略是主动把髋内滚到极限去并脚的。

而且**不只是站立**，每一档都窄（`scripts/measure_v03_stance.py`）：

| 档位 | 间距均值 | 比默认窄 |
|---|---|---|
| `stand` | 9.23 cm | 10.77 cm |
| `fwd` | 9.67 cm | 10.33 cm |
| `back` | 9.29 cm | 10.71 cm |
| `lat` | 12.94 cm | 7.06 cm |
| `turn` | 9.67 cm | 10.33 cm |

**为什么现成的 `pose` 没管住** —— 这是本节最值得记的一条：

`variable_posture` 的公式是 `exp(-mean(err²/std²))`，而站立档
`std_standing = {".*": 0.05}` 极紧。`hip_roll` 偏差 0.254 rad 时

```
(0.254 / 0.05)² = 25.8     →     exp(-25.8) ≈ 0
```

**该项在站立时已经彻底饱和**：不管策略把腿并得更窄还是张回正常，`pose`
都停在 0 附近，**梯度也归零**。策略早就不看它了，剩下各项的合力恰好偏好窄站姿。
所以「把 `std_standing` 再收紧一点」是**反效果**的 —— 越紧越早饱和。

**修法**：加一个**无界二次罚**，误差越大梯度越大：

```python
error = (|Δy_foot| − 0.20) / 0.05
reward = −0.25 × error²
```

量级：站立时偏差 10.8 cm → `(0.108/0.05)² = 4.67`，乘 −0.25 得 **−1.17**，
与 `pose`（权重 1.0）同量级，压得住；而步态内间距的自然波动 std 只有
~1 cm → 只罚 ~0.04，不干扰正常迈步。

**三个实现细节**（都在 `velocity/mdp/rewards.py::feet_lateral_distance`）：

1. **必须在 base heading 系量，不能在世界系量**。世界系 y 间距会随 yaw
   按 `cos(yaw)` 缩水 —— 机器人转到侧向时读数会假性接近 0，奖励自己就失效了。
   做法是 `quat_apply_inverse(yaw_quat(root_quat_w), Δpos_w)`，只去 yaw，
   保留俯仰/横滚（它们是姿态的一部分，不该被这个项抵消）。
2. **用 `|Δy|` 而非有符号差**，避免依赖 site 的解析顺序
   （`find_sites` 默认按模型顺序返回，不保证是传入顺序）。
   代价：双腿完全交叉时读数会假性正常 —— 那种姿态已由 `self_collisions` 罚。
3. **不加命令门控，全程生效**。因为实测每一档都窄，不是站立特有的问题；
   门控还会在站立↔行走切换处引入不连续。

> **一般教训**：`exp(-err²/std²)` 形式的奖励**在误差大时是死的**。
> 它适合做「跟踪」（最优解在 0 误差处，策略总在附近工作），
> 不适合做「约束」—— 约束一旦被违反得比较狠，奖励早就饱和了，
> 梯度消失，策略实际上获得了「随便违反」的自由。
> 约束类项要用**无界罚**（`err²`、`|err|`、huber）。

---

## 5 · 终止条件

| 项 | 参数 |
|---|---|
| `time_out` | episode 20 s |
| `fell_over` | `bad_orientation(limit_angle=70°)` |
| `out_of_terrain_bounds` | 仅 rough |

---

## 6 · 仿真

| 项 | flat | rough |
|---|---|---|
| `njmax` | 300 | 1500（骨架） |
| `nconmax` | None | 70 |
| `ccd_iterations` | 50 | 500 |
| `contact_sensor_maxmatch` | 64 | 500 |
| `timestep` | 0.005 | 0.005 |
| `decimation` | 4 | 4 |

→ 控制频率 **50 Hz**。

---

## 7 · PPO 超参（`rl_cfg.py`）

```python
actor/critic hidden_dims = (512, 256, 128), activation = "elu"
obs_normalization = True
GaussianDistribution(init_std = 1.0)

clip_param = 0.2          entropy_coef = 0.01
num_learning_epochs = 5   num_mini_batches = 4
learning_rate = 1.0e-3    schedule = "adaptive"
gamma = 0.99              lam = 0.95
desired_kl = 0.01         max_grad_norm = 1.0

num_steps_per_env = 24
save_interval = 500        # ← 每 500 次迭代存一次 pt
max_iterations = 30_000
experiment_name = "pikachu_v03_velocity"
```

观测维度：`actor Box(N, 50)` / `critic Box(N, 62)`（flat 版删掉了 `height_scan`）。

---

## 8 · 与 14dof / 27dof 的差异清单

1. 膝是**并联四杆**，只有 `knee_up` 驱动；`knee` / `knee_mid` 被动。
2. **无手臂** —— 动作维度 10（14dof 是 14）。
3. **左腿不是整体镜像** —— 只有 `ankle` / `knee_up` / `knee_mid` 反号。
4. 站姿 keyframe 是**静平衡点**（质心在脚心正上方），不是 qpos=0。
5. `hip_roll` / `hip_yaw` 限位仅 ±0.26 → 动作缩放收到 0.20。
6. 脚底是 **5 条胶囊**（V025 是 7 条脚趾胶囊）。
