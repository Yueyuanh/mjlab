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
rel_forward_envs = 0.25
rel_backward_envs = 0.25     # ← 显式给后退样本
rel_lateral_envs  = 0.25     # ← 显式给侧移样本
ranges.lin_vel_x = (-0.4, 0.5)
ranges.lin_vel_y = (-0.35, 0.35)
ranges.ang_vel_z = (-0.3, 0.3)
```

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
| `track_linear_velocity` | 2.0 | std = √0.25 | 骨架默认 |
| `track_angular_velocity` | 2.0 | std = √0.5 | 骨架默认 |
| `pose` | 1.0 | 见下 | **只覆盖 10 个驱动关节** |
| `upright` | 1.0 | `base_link` | — |
| `body_ang_vel` | **-0.05** | `base_link` | 抑制机身摇晃 |
| `angular_momentum` | **-0.02** | `root_angmom` | 抑制甩腿带来的角动量 |
| `base_vertical_velocity` | **-0.5** | — | 抑制上下弹跳 |
| `base_height` | **-0.2** | target 0.25 / std 0.02 | 防蹲防跳 |
| `air_time` | **+3.0** | `alternating_feet_air_time`，target 0.22 s / std 0.08 | **迈步的主要驱动** |
| `flight` | **-2.0** | grace 0.04 s | 禁止双脚同时离地（不是跑步任务） |
| `foot_clearance` | **-4.0** | target 0.04 m | 抬脚别太高也别拖地 |
| `foot_swing_height` | **-2.0** | target 0.04 m | 同上 |
| `soft_landing` | **-2e-3** | — | 落地缓冲 |
| `action_rate_l2` | **-0.15** | — | 动作平滑 |
| `action_acc_l2` | **-0.02** | — | 动作二阶平滑（新增项） |
| `dof_pos_limits` | -1.0 | — | 骨架默认 |
| `self_collisions` | **-1.0** | force_threshold 10 N | 新增项 |

### `pose` 的 std 表

```python
asset_cfg.joint_names = PIKACHU_V03_POSE_JOINT_NAMES   # 只 10 个驱动关节
std_standing = {".*": 0.05}
std_walking  = {hip_pitch 0.4, hip_roll 0.15, hip_yaw 0.15, knee_up 0.4, ankle 0.35}
std_running  = {hip_pitch 0.5, hip_roll 0.2,  hip_yaw 0.2,  knee_up 0.5, ankle 0.35}
running_threshold = 2.0
```

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
