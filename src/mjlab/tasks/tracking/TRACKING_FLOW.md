# Motion Tracking 任务流程梳理 · 数据集制作指南

> 适用任务：`Mjlab-Tracking-Flat-Unitree-G1`（包 `mjlab.tasks.tracking`）。
> 本文把 tracking 任务的**完整数据流**梳理清楚，重点回答"**如何为 Pikachu 制作参考动作数据集**"。
> 参考实现：BeyondMimic（[https://beyondmimic.github.io/](https://beyondmimic.github.io/)），
> 基于 [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)。

---

## 0. 为什么改用 tracking（背景）

当前 transition 任务是**隐式学习**起立：没有参考动作，只靠 `mode_pose` / `mode_upright` /
`transition_smoothness` 等奖励函数让策略自己"摸索"出站立↔趴下的过渡。难点在于：

- 过渡段是**非周期、一次性**的，奖励函数很难精确刻画"正确"的动作顺序；
- 纯 PD 拉 offset 做不了全幅姿态切换，策略需要极长时间的探索才能找到一条可行的起立路径；
- 结果：训练基本不收敛，"起立"学不出来。

**tracking（motion mimic）把"该做什么动作"从奖励函数中抽出来，变成一份参考动作数据**：
策略只负责"模仿"参考帧（关节角 + 各 body 位姿/速度），过渡段做成一段 motion clip（趴→站→趴）
之后，模仿即是起立。数据是外部的、可编辑的，不再依赖奖励函数试错。这是当前最可行的路径。

> 代价：你需要**先有一段参考动作数据**。数据质量直接决定模仿效果——这就是本文的重点。

---

## 1. 总体流程（一张图）

```
 数据源（CSV / MoCap / 手工关键帧）
        │  每行一帧：root_pos(3) + root_quat_xyzw(4) + dof_pos(14)
        ▼
 ┌─────────────────────────────────────────────┐
 │ scripts/csv_to_npz.py                       │
 │  ① 插值到输出 fps（位置/关节 lerp、四元数 slerp）│
 │  ② 差分求速度（线速度 / 关节速度 / SO(3) 角速度）│
 │  ③ 在 MuJoCo 里逐帧 playback + 校验          │
 │  ④ 导出 motion.npz（所有 body / 所有关节）      │
 └─────────────────────────────────────────────┘
        ▼
 motion.npz ──(W&B registry type="motions" 或本地路径)
        ▼
 MotionCommand（CommandManager）
   · MotionLoader 按 cfg.body_names 过滤 body 数组
   · sampling_mode: adaptive / uniform / start（决定起始帧）
   · reset 时把参考状态（带抖动）写入 sim
   · 每步 time_steps += 1，wrap 后重新采样；参考位姿按 robot root 重锚
        ▼
  观测（actor: 参考关节命令 + 锚点相对位姿 + IMU + 自身关节）
  奖励（BeyondMimic 指数误差：global root / relative body pos+ori / body vel）
  终止（锚点 z 误差 / 锚点朝向 / 末端 body z 误差）
        ▼
 训练（MotionTrackingOnPolicyRunner）──评估（evaluate.py：MPKPE/R-MPKPE…）
        ▼
 ONNX 导出（motion 数据一起打进模型，部署端按 time_step 同步播放参考）
```

---

## 2. 数据集格式：`motion.npz` 规格

`MotionLoader`（[mdp/commands.py](mdp/commands.py)）直接 `np.load(motion_file)` 读取，npz 内含：

| key | shape | 含义 |
|---|---|---|
| `fps` | `[1]` | 采样率，训练端未用（仅在制作端记录） |
| `joint_pos` | `[T, J]` | 每个关节的角度（**XML 关节序**，全部关节） |
| `joint_vel` | `[T, J]` | 关节角速度 |
| `body_pos_w` | `[T, NB, 3]` | **所有 body** 的世界系位置 |
| `body_quat_w` | `[T, NB, 4]` | **所有 body** 的世界系四元数（wxyz） |
| `body_lin_vel_w` | `[T, NB, 3]` | **所有 body** 的世界系线速度 |
| `body_ang_vel_w` | `[T, NB, 3]` | **所有 body** 的世界系角速度 |

三条关键约定：

1. **body 顺序 = XML 中 `<body>` 的出现顺序**（也就是 `robot.data.body_link_pos_w` 的列序）。
   制作端存的是"全部 body"（`robot.data.body_link_pos_w[0, :]`），训练端才按 `cfg.body_names`
   过滤。**body 索引 0 必须是 freejoint 所在的 root body**——`_resample_command` 用
   `self.body_pos_w[env_ids, 0]` 作为机器人 root 状态，Pikachu 即 `base_link`（XML 第一个 body）。
2. **joint 顺序 = XML 中 `<joint>` 的定义顺序**（`robot.data.joint_pos` 的列序）。
   注意 CSV 的 dof 列顺序由 `csv_to_npz.py` 的 `joint_names` 参数单独指定，可以任意，
   但建议与 XML 序一致，减少混乱。
3. **fps 必须与 env 的 step_dt 匹配**。tracking env 是 `timestep=0.005, decimation=4`
   → `step_dt = 0.02 s` = **50 fps**。制作端 `output_fps` 应设为 50，否则参考帧在训练时
   逐 `time_steps += 1` 推进的速度（每步一帧 = 50Hz）与数据集本身的运动节奏不一致。

---

## 3. 数据集制作全流程（`scripts/csv_to_npz.py`）

脚本地址：[scripts/csv_to_npz.py](../../scripts/csv_to_npz.py)（当前硬编码为 G1）。

### 3.1 输入 CSV 格式

每行一帧，列布局：

```
列 0–2   root_pos     (x, y, z)      base_link 的世界系位置
列 3–6   root_quat    (x, y, z, w)   ⚠️ xyzw 序！脚本加载后会转成 wxyz
列 7+    dof_pos      14 个关节角，按 joint_names 参数顺序
```

CSV 没有单位注释，位置为米、角度为弧度。

数据来源的几种方式（按可行性排序）：
- **手工关键帧插值**：蹲→撑手→趴（或反向）只需要 5–8 个关键帧，用 lerp/slerp 插出整段。
  起立这种过渡动作最推荐，完全可控。
- **实机/仿真录轨迹**：在 viewer 里手动控制或 PD 播放一段 demo，记录 root + 关节角成 CSV。
- **MoCap 重定向**：动捕数据重定向到 Pikachu（工作量大，但动作自然）。
- **Blender/动画软件导出**：关键帧动画导出 pose + 关节角。

> 手工关键帧起步：给出每个关键帧的 root_pos / root_quat / 14 关节角，然后线性插值到 30fps
> 左右存 CSV，再用 `csv_to_npz.py` 细化到 50fps。

### 3.2 脚本内处理管线

```
_load_motion()
  np.loadtxt → [:, :3]=root_pos, [:, 3:7]=root_quat(xyzw→wxyz), [:, 7:]=dof_pos
_interpolate_motion()
  input_fps → output_fps：位置/关节 lerp，四元数 slerp（quat_slerp）
_compute_velocities()
  base_lin_vel  = torch.gradient(root_pos, output_dt)
  dof_vel       = torch.gradient(dof_pos, output_dt)
  base_ang_vel  = SO(3) 差分（axis_angle(q_{t+1} * q_t^-1) / 2dt）
run_sim()   ← 关键步骤
  scene.reset()
  每帧：把 root_state + joint_pos/vel 写入 sim → sim.forward() → scene.update()
  校验：assert 仿真算出的 body0 线/角速度 ≈ 差分值（数据物理一致性门控）
  记录：robot.data 的 joint_pos/vel + 所有 body 的 pos/quat/lin_vel/ang_vel
  整段播完 → np.stack → np.savez("/tmp/motion.npz", **log)
  上传 W&B（type="motions"，registry），可选 render 视频
```

为什么要"在仿真里 playback"而不是直接用 CSV 差分：MuJoCo 前向动力学会对关节角/root 姿态做
内部修正（如四元数归一化、限位），`robot.data` 里的才是训练时观测到的真实量；且 body 的
位姿/速度全部由仿真算出，保证与训练 sim 一致。制作端还顺手 assert 了仿真速度与差分速度
的一致性，是数据质量的硬门控。

### 3.3 Pikachu 修改点（TODO，需手动改）

`csv_to_npz.py` 目前写死 G1，为 Pikachu 制作数据需改 `main()`：

```python
# ① scene 换 Pikachu：需要一个 Pikachu 的 tracking env cfg（见 §7 Checklist）
from mjlab.tasks.tracking.config.pikachu.env_cfgs import (
  pikachu_flat_tracking_env_cfg,  # 新建，复用 make_tracking_env_cfg()
)

scene = Scene(pikachu_flat_tracking_env_cfg().scene, device=device)

# ② joint_names 换 Pikachu 的 14 关节（必须与 CSV dof 列顺序一致）
joint_names = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
]

# ③ fps：Pikachu 用与 G1 相同的 tracking step_dt → output_fps=50
#    input_fps 按你的 CSV 实际 fps（30/50/60…）
```

Pikachu 的 XML body/joint 定义见
[asset_zoo/robots/pikachu/xmls/mjcf_14dof_lite/Pikachu_V025_flat_14dof_capsule.xml](
../../asset_zoo/robots/pikachu/xmls/mjcf_14dof_lite/Pikachu_V025_flat_14dof_capsule.xml)，
body 顺序以文件为准（base_link 在最前，其后双腿 5 关节链、双臂 2 关节链）。

### 3.4 数据质量检查

- `np.load("/tmp/motion.npz")` 查看各 key 的 shape：`T` 应 ≈ 数据秒数 × 50。
- 检查 `joint_pos` 是否全部落在 Pikachu 的软限位内
  （`robot.data.soft_joint_pos_limits`，可对照 `calibrate_quad_keyframe.py` 的输出）。
- 检查 root_z 是否合理：起立数据 base_link 的 z 应在 ~0.1（趴）→ ~0.32（站）之间变化。
- 制作端 assert 不报错 = 仿真速度与 CSV 速度一致，数据自洽。

### 3.5 常见坑

| 坑 | 现象 | 解决 |
|---|---|---|
| root_quat 用了 wxyz 而非 xyzw | 姿态错乱、插值异常 | CSV 必须是 **xyzw**（脚本 `[:, [3,0,1,2]]` 转 wxyz） |
| joint_names 与 CSV 列顺序不一致 | 关节角张冠李戴 | 严格一一对应，建议直接用 §3.3 的顺序 |
| output_fps ≠ 50 | 训练时动作节奏与参考帧推进速度不匹配 | `output_fps=50` |
| CSV 关节角超出限位 | reset 时被 clip，参考与真实不一致 | 用软限位裁剪后再存 |
| root_z 跳变 / 不连续 | 训练时 reset 写 sim 后瞬间穿地 | 关键帧插值要平滑；z 过低会触发 `bad_anchor_pos_z_only` |

---

## 4. 训练端加载与采样（`MotionCommand`）

`mdp/commands.py` 的 `MotionCommand` 是训练端核心：

- **`MotionLoader`**：`np.load` → 转 tensor → 按 `cfg.body_names`（`robot.find_bodies`）
  过滤 body 数组，只保留关心 body。`time_step_total = T`。
- **起始帧采样**（`sampling_mode`，见 `MotionCommandCfg`）：
  - `adaptive`（默认）：把整段 motion 分成 `bin_count` 个 bin，每个 bin 记录历史失败率
    （EMA，`adaptive_alpha=0.001`），失败多的段被重采样概率越高（困难优先学习）。
    用非因果卷积核平滑 + `adaptive_uniform_ratio=0.1` 均匀保底。
  - `uniform`：随机起始帧。
  - `start`：固定帧 0（play / evaluate 用）。
- **`reset` / `_resample_command`**：采样起始帧 → 取参考 root pos/quat/lin_vel/ang_vel + 
  joint pos/vel → 叠加 `pose_range`（x/y/z/roll/pitch/yaw 抖动）、`velocity_range`、
  `joint_position_range` 抖动 → `_write_reference_state_to_sim` 写入 sim（关节裁剪到软限位）。
- **每步 `_update_command`**：`time_steps += 1` → 越过末尾则重新采样（wrap）→
  `sim.forward()` → `update_relative_body_poses()`。
- **`update_relative_body_poses`**：以 robot 当前 root（锚 body）为基准，把参考 body 位姿
  **重锚**（平移取 robot 锚的 z、朝向只取 yaw 对齐），得到 `body_pos_relative_w` /
  `body_quat_relative_w`——这是"相对位姿"奖励与 R-MPKPE 指标的量，剔除了全局漂移，
  只比内在姿态。
- **`command`**：`cat([joint_pos, joint_vel])`，作为观测里的参考通道，让策略"看到"接下来
  该往哪动。

`MotionCommandCfg` 需要显式给的字段：`motion_file`、`anchor_body_name`（锚/根 body）、
`body_names`（追踪的 body 列表，**第一个不必是锚**）、`entity_name`，以及各类 `*_range`。

---

## 5. MDP：观测 / 奖励 / 终止

### 观测（[mdp/observations.py](mdp/observations.py)）

- **actor**：`command`（参考 joint_pos+vel，2J 维）、`motion_anchor_pos_b`（参考锚相对 robot
  锚，body 系 3 维）、`motion_anchor_ori_b`（相对旋转矩阵前两列 6 维）、`base_lin_vel` /
  `base_ang_vel`（IMU）、`joint_pos_rel` / `joint_vel_rel`、`actions`。
- **critic 追加**：`body_pos`（全部追踪 body 相对锚的 pos）、`body_ori`。

> `motion_anchor_*_b` 给 actor 提供"参考要往哪走"的全局线索（body 系），配合参考关节命令，
> 策略不需要状态估计也能跟随（`has_state_estimation=False` 变体就是去掉它测的）。

### 奖励（[mdp/rewards.py](mdp/rewards.py)，全部是指数误差 `exp(-err/std²)`）

| 名称 | 比较量 | 权重 | std |
|---|---|---|---|
| `motion_global_root_pos` | anchor 世界位置 | 0.5 | 0.3 |
| `motion_global_root_ori` | anchor 四元数误差 | 0.5 | 0.4 |
| `motion_body_pos` | **relative** body 位置（重锚后） | 1.0 | 0.3 |
| `motion_body_ori` | **relative** body 朝向 | 1.0 | 0.4 |
| `motion_body_lin_vel` | body 世界线速度 | 1.0 | 1.0 |
| `motion_body_ang_vel` | body 世界角速度 | 1.0 | 3.14 |
| `action_rate_l2` | 动作平滑 | −0.1 | — |
| `joint_limit` | 关节限位 | −10.0 | — |
| `self_collisions` | 自碰撞力 | −10.0 | — |

> 权重最大的是 **relative body pos / ori**（各 1.0）：它们是"模仿姿态"的主体，root 重锚后
> 不惩罚全局漂移（漂移交给 global root 项，权重 0.5）；速度项确保动作节奏一致。
> 对 Pikachu 起立，应重点保证这段数据里 body（尤其手/脚/躯干）相对位姿变化正确。

### 终止（[mdp/terminations.py](mdp/terminations.py)）

| 名称 | 条件 |
|---|---|
| `time_out` | episode 时长到 |
| `bad_anchor_pos_z_only` | 锚 body z 与参考差 > 0.25 m（摔地/飞起） |
| `bad_anchor_ori` | 锚投影重力与参考差 > 0.8（摔倒） |
| `bad_motion_body_pos_z_only` | 任一指定末端 body 的 z 与参考差 > 0.25 m（如脚/手脱离参考） |

---

## 6. 训练 / 评估 / 部署命令

```sh
# 训练（motion 数据两种给法）：
#   ① 本地 npz
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
    --env.commands.motion.motion-file /path/to/motion.npz \
    --env.scene.num_envs=4096
#   ② 从 W&B registry 下载（type="motions"）
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
    --registry-name your-org/motions/motion-name \
    --env.scene.num_envs=4096

# 网页端观察（ghost 显示参考、frames 显示坐标系，可拖动 scrubber）
uv run play Mjlab-Tracking-Flat-Unitree-G1 --viewer viser --checkpoint-file logs/.../model_xxx.pt

# 评估（MPKPE / R-MPKPE / joint_vel_error / ee_* / success_rate）
uv run python src/mjlab/tasks/tracking/scripts/evaluate.py \
    --wandb-run-path entity/project/run_id
```

`train.py` 通过 `commands["motion"]` 是否为 `MotionCommandCfg` 自动识别 tracking 任务；
`--env.commands.motion.motion-file` 指定本地数据，否则要求 `--registry-name`。

ONNX 导出（`MotionTrackingOnPolicyRunner.export_policy_to_onnx`）会把整份 motion 数据
（joint_pos/vel、body_*）作为 buffer 打进模型，输入 `(obs, time_step)`，输出动作 +
当前参考帧——部署端不需要单独加载 npz，按 `time_step` 播放即可。

---

## 7. Pikachu 落地 Checklist（TODO，需手动做）

1. **制作 CSV 参考数据**：先做一小段"趴→站→趴"（或你要学的过渡）的关键帧，插值成 CSV
   （root_pos + root_quat_xyzw + 14 关节角，见 §3.1）。
2. **新建 Pikachu tracking env cfg**：`tasks/tracking/config/pikachu/env_cfgs.py`，
   复用 `make_tracking_env_cfg()`，设置：
   - `cfg.scene.entities = {"robot": get_pikachu_robot_cfg()}`（用 `pikachu_constants.py`）
   - `motion_cmd.anchor_body_name = "base_link"`（freejoint root body）
   - `motion_cmd.body_names = (...)`（建议：base_link + 双腿 hip/knee/ankle + 双手 arm_yaw，
     共 11 个，见下）
   - `motion_cmd.motion_file = "/path/to/motion.npz"`
   - `cfg.terminations["ee_body_pos"].params["body_names"] = (left/right_ankle_link, left/right_arm_yaw_link)`
   - `cfg.events["foot_friction"].params["asset_cfg"].geom_names`、`base_com` body_names 按
     Pikachu 实际填（对照 velocity 的 Pikachu 配置）
   - `cfg.viewer.body_name = "base_link"`
3. **body_names 建议**（锚 + 关键 body，控制数据体积与关注点）：
   ```python
   (
     "base_link",
     "left_hip_pitch_link",
     "left_knee_link",
     "left_ankle_link",
     "right_hip_pitch_link",
     "right_knee_link",
     "right_ankle_link",
     "left_arm_pitch_link",
     "left_arm_yaw_link",
     "right_arm_pitch_link",
     "right_arm_yaw_link",
   )
   ```
   `arm_yaw_link` 是手（四足撑地点），起立数据里手的位置/着地时机是核心，一定要追踪。
4. **改 `csv_to_npz.py`** 的 scene + joint_names + output_fps（§3.3），跑通制作并检查 npz。
5. **注册任务** `Mjlab-Tracking-Flat-Pikachu`（`config/pikachu/__init__.py`，仿 G1），
   配 `rl_cfg`（可先复用 G1 的 512/256/128 + ELU）。
6. **训练冒烟**：`--env.scene.num_envs=1024 --agent.max_iterations=500`，确认奖励上升、
   参考不 collapse、play 里 ghost 能对上。
7. **调参入口**：`sampling_mode`（起立是单段过渡，建议先从 `uniform` 或短 clip 开始）、
   `pose_range` / `velocity_range`（过大 reset 会穿地）、body 相关 std、termination 阈值。

---

## 8. 与 transition 任务的关系（可选）

tracking 学好"起立"后，可以：
- 把 tracking 策略学到的起立动作**导出为 motion clip**，再作为 transition 的参考；
- 或者直接在 tracking 框架里做"双足走路 + 起立 + 四足爬行"的拼接数据集，一次学到全部
  模式与过渡（BeyondMimic 正是这么做的），替代 transition 的隐式奖励方案。

> 两种路线的取舍：纯 tracking 需要每一段行为都有数据（走路也要录）；transition 的
> 隐式奖励对**稳态**（走/趴）仍然有效。折中：tracking 学过渡段，velocity/transition 学稳态，
> 再融合。
