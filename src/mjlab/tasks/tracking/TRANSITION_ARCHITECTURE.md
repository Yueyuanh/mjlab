# Pikachu 状态转换 · 三模型架构（tracking 只做过渡段）

> **一句话**：把"行走 / 爬行 / 过渡"拆成**三个独立模型**，各用最擅长的训练方式——
> 稳态（走、爬）是周期性行为，用速度跟踪类任务（velocity）训；过渡段（站↔趴）是
> 一次性非周期行为，奖励函数刻不准但动作内容明确，用**参考动作模仿（tracking）**训。
> 上层用规则状态机切换三个模型。
>
> 关联文档：数据集制作细节见 [TRACKING_FLOW.md](TRACKING_FLOW.md)（§2–3 数据集格式/制作，
> §5 观测奖励）。本架构是对 transition 单策略方案的替代/演进。

---

## 1. 为什么不用"单策略 + mode 命令"了

transition 任务的单策略方案（`Mjlab-Transition-Flat-Pikachu`）让一个策略同时学两态步态 +
隐式过渡，实践下来问题集中在过渡段：

- 起立/趴下是**一次性、非周期**动作，`mode_pose` / `mode_upright` / `transition_smoothness`
  这些奖励函数描述不了"正确的动作顺序"，策略只能靠探索碰运气，基本不收敛；
- 稳态（走/爬）与过渡耦合在同一个优化里，两态步态互相干扰；
- 纯 PD 拉 offset 也完成不了全幅姿态切换（见 transition 任务 RISK R-16）。

**拆分后**：稳态模型收敛快、已有一个能用（行走）；过渡段不再靠奖励试错，而是**直接模仿
一段参考动作**——数据是外部的、可编辑的，起立能不能学会取决于数据质量，不取决于奖励函数。

---

## 2. 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                    上层 Mode 状态机（规则驱动）                 │
│  当前模式 ∈ {双足稳态, 四足稳态, 过渡中(b→q), 过渡中(q→b)}      │
│  触发：外部指令（手动/任务） → 进入过渡 → 激活 tracking 过渡策略 │
│  完成判定：g_b 夹角 + 手脚接触稳定 或 clip 播完 → 切目标稳态策略 │
└──────────────────────────────────────────────────────────────┘
                    │激活          │激活              │激活
        ┌───────────▼───┐  ┌───────▼───────┐  ┌───────▼───────┐
        │ 双足行走模型    │  │ 四足爬行模型    │  │ 过渡模型(×2方向)│
        │ velocity 任务   │  │ transition→四足 │  │ tracking 任务  │
        │ BIPED_KEYFRAME │  │ QUAD_KEYFRAME  │  │ 过渡 clip 数据 │
        │ twist 速度跟踪  │  │ +手接触奖励     │  │ 参考动作模仿    │
        └────────────────┘  └────────────────┘  └───────────────┘
          走（周期性稳态）      爬（周期性稳态）       站↔趴（一次性）
```

三个模型**训练完全独立**，互不依赖（过渡模型只依赖数据集，不依赖另两个策略）。

---

## 3. 三个模型定义

### 3.1 行走模型（双足稳态）—— 已有，不动

| 项 | 值 |
|---|---|
| 任务 | `Mjlab-Velocity-Flat-Pikachu`（velocity 任务，已验证） |
| 初始姿态 | `BIPED_KEYFRAME`（= KNEES_BENT，微蹲起跑） |
| 观测/奖励 | velocity 全家桶（twist 速度跟踪 + 脚部奖励） |
| 碰撞 | `FEET_ONLY_COLLISION`（脚-地接触） |

### 3.2 爬行模型（四足稳态）—— 改造 transition 任务

| 项 | 值 |
|---|---|
| 任务 | **改造 `Mjlab-Transition-Flat-Pikachu` → 纯四足 velocity 变体**（推荐，见 §7） |
| 初始姿态 | `QUAD_KEYFRAME`（base 前俯 90° + 手/脚四点撑地，几何已验证） |
| 观测 | velocity 观测 + 手接触 `hand_ground_contact`（已有传感器） |
| 奖励 | velocity 速度跟踪 + `quad_hand_contact` / `quad_hand_slip`（已有）+ 四足姿态 std |
| 碰撞 | `FULL_COLLISION_WITHOUT_SELF`（手 `arm_yaw_link` condim=3 + friction，防打滑） |
| 命令 | 只有 `twist`，**没有** mode 命令（mode 固定 = 1） |

关键点：transition 任务已经把四足需要的**几何、手传感器、手部奖励、碰撞配置**全部做好了，
改造 = **删掉 mode 命令 / 双足分支 / 切换相关**，保留四足分支，命令换成纯 twist。这样爬行
就是"四足版的 velocity 任务"，收敛快、可长期优化。

### 3.3 过渡模型（站↔趴）—— tracking，本架构的核心新增

| 项 | 值 |
|---|---|
| 任务 | `Mjlab-Tracking-Flat-Pikachu`（新建，复用 `make_tracking_env_cfg()`） |
| 数据 | **过渡 clip × 2 方向**：`biped→quad`（趴下）、`quad→biped`（站起），各 2~3 s @ 50 fps |
| 初始姿态 | 每段 clip 的起始帧（推理时不瞬移：首帧 ≈ 当前稳态姿态） |
| 观测 | 参考关节命令(joint_pos+vel) + 锚点相对位姿(`motion_anchor_pos_b/ori_b`) + IMU + 自身关节 |
| 奖励 | BeyondMimic 指数误差，以 **relative body pos/ori**（各 1.0）为主（见 TRACKING_FLOW §5） |
| 锚点 | `anchor_body_name = "base_link"`（freejoint root body） |
| body 追踪 | base_link + 双腿 hip/knee/ankle + 双手 `arm_yaw_link`（手必须追踪，是撑地与起立的关键） |

两个方向**拆成两个模型**（两个 checkpoint），不要拼成一段：趴下和站起动作模式完全不同，
拼成一段还需要处理中间无法硬接的过渡帧。命名如
`Mjlab-Tracking-Flat-Pikachu-Down` / `Mjlab-Tracking-Flat-Pikachu-Up`。

---

## 4. 过渡数据集（关键路径）

这是整个方案里**唯一的硬前置**，也是数据制作的重点（详见 TRACKING_FLOW.md §2–3）。

需要的两段 clip：

| clip | 起点姿态 | 终点姿态 | 关键帧建议（手工插值起步） |
|---|---|---|---|
| `down`（biped→quad） | 站立 ≈ `BIPED_KEYFRAME` | 趴下 ≈ `QUAD_KEYFRAME` | 站立 → 屈膝下蹲 → 手前伸触地 → 重心前移手撑地 → 躯干俯下成四点支撑 |
| `up`（quad→biped） | 趴下 ≈ `QUAD_KEYFRAME` | 站立 ≈ `BIPED_KEYFRAME` | 趴下 → 手撑地撑起躯干 → 重心后移脚撑地 → 屈膝站起 → 直立 |

首帧/末帧要求：
- `down` 的**首帧 = 行走结束姿态**，`up` 的**末帧 = 行走起始姿态**（→ 能接上行走模型）
- `up` 的**首帧 = 爬行结束姿态**，`down` 的**末帧 = 爬行起始姿态**（→ 能接上爬行模型）
- 这样推理时切换瞬间两策略输出 ≈ 同一姿态，动作连续。

制作管线（引用 TRACKING_FLOW.md §3）：
```
手工关键帧(5~8帧: root_pos + root_quat_xyzw + 14关节角)
   → 插值成 CSV(30fps) → csv_to_npz.py(重采样到50fps + 仿真playback校验) → motion.npz
   → 注册 W&B registry(type="motions") 或本地路径 → tracking 训练
```

---

## 5. 推理分层控制

### 5.1 状态机

```
   BIPED ──指令"四足"──▶ TRANSITION(b→q) ──完成判定──▶ QUAD
      ▲                                                   │
      └───────────────────────────────────────────────────┘
   QUAD ──指令"双足"──▶ TRANSITION(q→b) ──完成判定──▶ BIPED
```

- 状态机是**规则逻辑**（不是 RL 策略），部署在推理层，读 IMU/关节/接触判定模式。
- 过渡中收到反向指令：优先让当前过渡走完（或从当前位置反向换策略，属 TODO）。

### 5.2 完成判定（过渡结束 → 切稳态模型）

- `projected_gravity_b` 夹角接近目标：双足 (0,0,-1)、四足 (+1,0,0)（QUAD 几何已验证），
  夹角 < 阈值（如 30°，可调）；
- 支撑稳定：目标稳态的脚（+ 四足时双手）接触力 > 0；
- 稳定持续 N 帧（如 10 帧）；
- 兜底：clip 播完（clip 只有 2~3 s，最坏情况多等几帧）。

### 5.3 过渡衔接的连续性（本架构核心难点，必须处理）

切换瞬间**策略换、观测通道换**（velocity 观测 = twist + 自身状态；tracking 观测 =
参考关节命令 + 锚点相对位姿 + 自身状态）。连续性的保证方式：

1. **clip 首帧 = 当前稳态姿态**（§4 的首帧要求）→ tracking 首帧"参考 = 自身"，
   模仿误差 ≈ 0 → 动作增量 ≈ 0，与上一策略输出接近，动作连续；
2. **动作通道对齐**：三模型都是 `JointPositionAction`（输出关节目标角），维度都是 14，
   没有维度跳变问题；只要求 clip 首/末帧关节角 ≈ 稳态姿态关节角；
3. **观测连续性**：tracking 观测里的"自身状态"部分（IMU/joint/actions）与 velocity 连续；
   参考通道在切换瞬间对应稳态姿态，无跳变；
4. **反向接管**：clip 末帧 = 目标稳态姿态 → 切换时稳态模型输出 ≈ 稳态 → 连续。

> 训练时 tracking reset 会把 sim 姿态瞬移到 clip 首帧（并叠加 `pose_range`/`velocity_range`
> 抖动）——这是训练常规做法，用于增强对"从各种近似姿态起播"的鲁棒性；推理时不需要瞬移，
> 因为首帧就是当前姿态，直接开播即可。

### 5.4 与现有 GUI 的关系

transition 任务现有的 `ModeCommand.create_gui`（viser 网页端 Manual Mode + Biped/Quad 按钮）
可**原样复用**作为上层状态机的触发入口：按钮 → 状态机 → 激活对应模型，而不是改 mode 命令。

---

## 6. 训练计划

| 顺序 | 模型 | 做什么 | 依赖 |
|---|---|---|---|
| 1 | 行走 | 已有 `Mjlab-Velocity-Flat-Pikachu`，无需重训 | — |
| 2 | 爬行 | 改造 transition → 纯四足 velocity，训定态步态 | 不需要数据集 |
| 3a | 过渡 down | **先做 `down` 数据集** → 训 tracking | 数据集（关键路径） |
| 3b | 过渡 up | **再做 `up` 数据集** → 训 tracking | 数据集 |

三模型相互独立，2 可与 3a/3b 并行（3 依赖数据集制作）。

---

## 7. 需要拍板的决策点（TODO）

1. **爬行模型的载体**（推荐 A）：
   - **A（推荐）**：改造 `Mjlab-Transition-Flat-Pikachu` → 删 mode 命令/双足/切换，保留
     四足分支（手传感器/手奖励/QUAD 几何/碰撞全复用），命令改纯 twist。改动量小、复用最多。
   - **B**：`Mjlab-Velocity-Flat-Pikachu` 直接换 `QUAD_KEYFRAME` 初始 + 加手传感器/手奖励。
     更干净但要从头配手部奖励（transition 里已有现成项）。
2. **过渡 clip 数据源**：手工关键帧插值起步（推荐，可控、快），MoCap/实机录作为后续。
3. **两方向是否合一**：推荐分开（两个模型），已在 §3.3 说明理由。
4. **上层状态机**：规则状态机（推荐），触发源 = 手动指令（复用 GUI）；是否需要"自动
   模式识别"（无指令自行判断当前模式）属后续。
5. **完成判定阈值**：g_b 夹角 / 支撑稳定帧数，viewer 实测后微调。

---

## 8. 落地顺序（供后续执行）

1. 改造 transition → 纯四足爬行任务（§3.2），训出爬行模型；
2. 做 `down` / `up` 两段过渡数据集（§4，关键路径）；
3. 建 tracking 任务 `Mjlab-Tracking-Flat-Pikachu-Down/Up`（§3.3），训出两个过渡模型；
4. 上层状态机 + GUI 复用（§5），串联三模型在 viewer 里验证完整状态转换；
5. 实机部署：三模型各自 ONNX 导出 + 状态机逻辑。

> 注意：tracking 与 velocity/transition 的观测/动作通道不同，第 4 步的"切换逻辑"是衔接
> 三模型的关键，具体实现时以 §5.3 的连续性要求为验收标准（切换瞬间动作/姿态无跳变）。
