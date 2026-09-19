# Pikachu / mjlab 任务谱系表

本表列出本 fork 中全部已注册任务，用于判断「当前主路径」「历史实验」「上游示例」。
维护约定：新增任务时在此表加一行；废弃任务改 Status 并保留 Task ID（勿静默删除注册）。

字段说明：

- **Status**：`active`（当前主路径，优先训练） / `experiment`（专项实验） / `legacy`（已被后续构型取代，保留复现） / `upstream`（上游示例） / `blocked`（框架就绪但缺关键输入）
- **Robot**：机器人 / 构型 / hip 轴序
- **Policy joints**：策略动作驱动的关节组
- **Config path**：配置包路径（相对 `src/mjlab/tasks/`）
- **Pretrain**：`pre_train/rsl_rl/` 下对应部署 run（若有）

---

## 1. 机器人构型演进

```
capsule 14-dof          sample-links v2.1           sample-links v2.2          27-dof sample-links
(pikachu / pikachu_quad) (pikachu_sl_v21_*)          (pikachu_sl_v22_*)         (pikachu_27dof*)
        │                        │                          │                          │
        │                        │                          │                          ├─ PRY hip（默认训练体）
        │                        │                          │                          └─ YRP hip（pikachu_27dof_yrp*，WIP）
        │                        │                          │
        └── 双足 / 四足 ─────────┴── 双足 walk / 转换 ──────┴── 四足 velocity ──────── 双足 / 四足 / transition
```

| 代际 | 标识 | 说明 |
|------|------|------|
| v0 capsule | `pikachu`, `pikachu_quad`, `pikachu_transition`, `pikachu_walk` | 早期胶囊碰撞体，14-dof 左右 |
| v2.1 sample-links | `pikachu_sl_v21_*` | 拟真连杆几何第一代 |
| v2.2 sample-links | `pikachu_sl_v22_*` | 比例/碰撞更新后的拟真连杆 |
| 27-dof | `pikachu_27dof*` | 全身 27 关节（腿10+臂8+头3+耳4+尾2） |
| 27-dof YRP | `pikachu_27dof_yrp*` | 同名 body，hip 拆分改为 Yaw→Roll→Pitch；参考 npz 运行时按名重排 |

---

## 2. Velocity 任务

| Task ID | Robot / 构型 | Policy joints | Config | W&B project | Status | Pretrain |
|---------|--------------|---------------|--------|-------------|--------|----------|
| `Mjlab-Velocity-Flat-Pikachu` | capsule 双足 | 腿 | `velocity/config/pikachu/` | `pikachu-velocity-flat` | legacy | — |
| `Mjlab-Velocity-Rough-Pikachu` | capsule 双足 + 地形 | 腿 | `velocity/config/pikachu/` | `pikachu-velocity-rough` | legacy | — |
| `Mjlab-Velocity-Flat-Pikachu-Quad` | capsule 四足 | 腿10 + 臂前腿6 | `velocity/config/pikachu_quad/` | `pikachu-velocity-quad-flat` | legacy→experiment | `.../trial-quad-1` |
| `Mjlab-Velocity-Flat-Pikachu-Quad-Bounding` | capsule 四足 bounding | 前手/后脚有序奖励 | `velocity/config/pikachu_quad_bounding/` | `pikachu-velocity-quad-bounding` | experiment | — |
| `Mjlab-Velocity-Flat-Pikachu-Sl22-Quad` | v2.2 四足 | 腿 + 前腿 | `velocity/config/pikachu_sl_v22_quad/` | `pikachu-sl22-velocity-quad-flat` | experiment | — |
| `Mjlab-Velocity-Flat-Pikachu-27dof` | 27-dof 双足 (PRY) | 腿10；上肢17为随机摆动干扰 | `velocity/config/pikachu_27dof/` | `pikachu-27dof-velocity-flat` | **active** | `full_27dof_*`, `27dof_zero_stand` |
| `Mjlab-Velocity-Flat-Pikachu-27dof-Quad` | 27-dof 四足 (PRY) | 腿10+臂6=16 | `velocity/config/pikachu_27dof_quad/` | `pikachu-27dof-velocity-quad-flat` | **active** | — |
| `Mjlab-Velocity-Flat-Pikachu-27dof-Yrp` | 27-dof 双足 (YRP) | 同 27dof 双足 | `velocity/config/pikachu_27dof_yrp/` | `pikachu-27dof-yrp-velocity-flat` | experiment (WIP untracked) | — |

上游示例（非 Pikachu）：`Mjlab-Velocity-{Flat,Rough}-Unitree-Go1`、`Mjlab-Velocity-{Flat,Rough}-Unitree-G1`。

---

## 3. Tracking（动作模仿）任务

| Task ID | Robot / 构型 | Policy joints | Config | W&B project | Status | Pretrain |
|---------|--------------|---------------|--------|-------------|--------|----------|
| `Mjlab-Tracking-Flat-Pikachu-Down` | capsule 双足→四足 | 跟踪 transition | `tracking/config/pikachu_transition/` | `pikachu-transition-down` | legacy | `trial-pikachu-transition-down-dual` |
| `Mjlab-Tracking-Flat-Pikachu-Up` | capsule 四足→双足 | 跟踪 transition | `tracking/config/pikachu_transition/` | `pikachu-transition-up` | legacy | `trial-pikachu-transition-up-dual` |
| `Mjlab-Tracking-Flat-Pikachu-Walk` | capsule 双足前进 | 腿+臂14 | `tracking/config/pikachu_walk/` | `pikachu-walk-forward` | legacy | `trial-pikachu-walk-forward-dual` |
| `Mjlab-Tracking-Flat-Pikachu-Walk-Base` | capsule 双足消融 | 纯腿10；臂为正弦扰动 | `tracking/config/pikachu_walk_base/` | `pikachu-walk-base` | experiment | `trial-pikachu-walk-base-arm-ablated` |
| `Mjlab-Tracking-Flat-Pikachu-Sl21-Down` | v2.1 双→四 | transition | `tracking/config/pikachu_sl_v21_transition/` | `pikachu-sl21-transition-down` | legacy | — |
| `Mjlab-Tracking-Flat-Pikachu-Sl21-Up` | v2.1 四→双 | transition | `tracking/config/pikachu_sl_v21_transition/` | `pikachu-sl21-transition-up` | legacy | — |
| `Mjlab-Tracking-Flat-Pikachu-Sl21-Walk` | v2.1 双足前进 | 腿+臂 | `tracking/config/pikachu_sl_v21_walk/` | `pikachu-sl21-walk-forward` | legacy | — |
| `Mjlab-Tracking-Flat-Pikachu-Sl22-Walk` | v2.2 双足前进 | 腿+臂 | `tracking/config/pikachu_sl_v22_walk/` | `pikachu-sl22-walk-forward` | experiment | `full_Walk_*` |
| `Mjlab-Tracking-Flat-Pikachu-27dof-Walk` | 27-dof 双足 (PRY) | 腿10；上肢随机摆动 | `tracking/config/pikachu_27dof/` | `pikachu-27dof-walk` | **active** | `trial_pikachu_27dof_walk_forward` |
| `Mjlab-Tracking-Flat-Pikachu-27dof-Quad-Walk` | 27-dof 四足 (PRY) | 腿10+臂6=16 | `tracking/config/pikachu_27dof_quad/` | `pikachu-27dof-quad-walk` | **blocked**（参考 npz 曾为占位；有 pretrain） | `quad_tracking` |
| `Mjlab-Tracking-Flat-Pikachu-27dof-Yrp-Walk` | 27-dof 双足 (YRP) | 同 27dof walk | `tracking/config/pikachu_27dof_yrp/` | `pikachu-27dof-yrp-walk` | experiment (WIP untracked) | — |
| `Mjlab-Tracking-Flat-Pikachu-27dof-Yrp-Quad-Walk` | 27-dof 四足 (YRP) | 同 27dof quad | `tracking/config/pikachu_27dof_yrp_quad/` | `pikachu-27dof-yrp-quad-walk` | experiment (WIP untracked) | — |

上游示例：`Mjlab-Tracking-Flat-Unitree-G1`、`Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation`。

---

## 4. 其它任务

| Task ID | 类型 | Config | Status |
|---------|------|--------|--------|
| `Mjlab-Cartpole-Balance` | 框架自检 | `cartpole/` | upstream |
| `Mjlab-Cartpole-Swingup` | 框架自检 | `cartpole/` | upstream |
| `Mjlab-Lift-Cube-Yam` | 操作 | `manipulation/config/yam/` | upstream |
| `Mjlab-Lift-Cube-Yam-Rgb` | 操作 + RGB | 同上 | upstream |
| `Mjlab-Lift-Cube-Yam-Depth` | 操作 + Depth | 同上 | upstream |
| `Mjlab-Multi-Cube-Seg-Yam` | 多物体 | 同上 | upstream |

---

## 5. 当前推荐训练入口

| 目标 | 命令 |
|------|------|
| 27-dof 双足 velocity（主路径） | `uv run train Mjlab-Velocity-Flat-Pikachu-27dof --env.scene.num-envs 512` |
| 27-dof 双足 walk tracking（主路径） | `uv run train Mjlab-Tracking-Flat-Pikachu-27dof-Walk --env.scene.num-envs 256` |
| 27-dof 四足 velocity | `uv run train Mjlab-Velocity-Flat-Pikachu-27dof-Quad --env.scene.num-envs 512` |
| 网页启动台 | `uv run start_server` |
| 冒烟（两主任务） | `bash scripts/train_pikachu_27dof.sh smoke` |

标定提醒：四足类任务的 `tilt_deg` / `base_height` 需先 `--agent zero` 在 viewer 实测回填（见各 `env_cfgs.py` 内 TODO）。

---

## 6. 相关文档

- 训练 / 调参流程：`TrainSpec.md`
- 分支与 W&B 约定：`CLAUDE.md`
- 参考动作与 PD 录制：`src/mjlab/mocap/README.md`
- 执行器响应测试：`src/mjlab/response/README.md`
- 部署 corpus 政策：`pre_train/README.md`