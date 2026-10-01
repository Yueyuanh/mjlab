# 04 · 工作日志

> 按时间**倒序**。每条记「做了什么 / 为什么 / 结论」。

---

## 2026-10-01 · 资产接入 + 静平衡标定（完成）

### 目标
把 `pikachu_v03_base_series` 接入 mjlab，新建 velocity 任务，训到稳定行走。

### 环境约束（先摸清楚，后面所有决策都受它影响）
- 内存 14 GB，可用约 5~6 GB，**swap 已满** → 同一进程里建两个 mjlab 环境
  会 OOM 被 SIGKILL（退出码 137）。
- 磁盘 468 GB **已用 93%，剩 35 GB** → 需要持续盯。
- GPU 8 GB。

### 完成项

**1. 分支**
```bash
git checkout -b task/pikachu-v03-locomotion
```

**2. 训练 MJCF 生成** — `scripts/build_v03_train_xml.py`
源模型（`pikachu_urdf/pikachu_v03_base_series/`，含 meshes）→
`src/mjlab/asset_zoo/robots/pikachu/xmls/mjcf_v03/`。
改造 6 处（freejoint / geom 命名 / 足部 site+胶囊 / 删 scene include /
删 actuator / 补 sensor），详见 [01-model-conversion.md](01-model-conversion.md)。

验收：`nq=21 nv=20 nu=0 njnt=15 nbody=16 ngeom=40`，足部胶囊 10 个。

**3. 双腿独立解 + 静平衡固定点迭代** —
`scripts/probe_v03_both_legs.py`、`scripts/refine_v03_keyframe.py`
产出 `KNEES_BENT_KEYFRAME`，详见 [02-stand-keyframe.md](02-stand-keyframe.md)。
顺带实测确认了**左腿不是整体镜像**（只有 ankle / knee_up / knee_mid 反号）。

**4. velocity 任务** — `src/mjlab/tasks/velocity/config/pikachu_v03/`
注册 `Mjlab-Velocity-Flat-Pikachu-V03` / `Mjlab-Velocity-Rough-Pikachu-V03`
（`uv run list-envs` 确认为 #24 / #25）。
详见 [03-task-design.md](03-task-design.md)。

**5. `pikachu_metrics.py` 参数化** — 支持 v03 的 10 关节表与电机分组，
14dof 任务走默认值不受影响。

**6. 冒烟测试** — `scripts/smoke_v03_velocity.py`（分 `--quiet` / `--noisy`
两个进程跑，绕开 OOM）

| 检查项 | 结果 |
|---|---|
| 动作维度 | 10 ✓ |
| 关节数 | 14 ✓ |
| 执行器数 | 10 ✓ |
| 被动关节 | `knee` / `knee_mid` ×2 ✓ |
| IMU 帧一致性 | 偏差 **0.000e+00** ✓ |
| 观测空间 | `actor Box(N,50)` / `critic Box(N,62)` |
| **阶段 A 零动作开环站立** | base_z 0.26728→0.26731，\|v\|→0.00000，6 s 无终止 **✓ 通过** |
| 阶段 B 完整扰动 | 最低 0.177 m，从未跌破半高，无终止 |

### 关键发现（都是踩过的坑）

1. **左腿不是整体镜像**。按直觉写 `left_ankle = -0.5148` 是错的，
   实测是 `+0.5048`。写 keyframe 前必须先打印 `jnt_axis` / `jnt_range`。
2. **源模型必须删 `<actuator>`**，否则 `nu` 翻倍（mjlab 只追加不清理）。
3. **必须补 `<sensor>` 块**，且 **imu site 必须是单位四元数**
   （14dof 资产那个 107° 的 quat 不要照抄）。
4. **「脚在髋下」≠「静平衡」**。v03 质心在 base 原点后方 2 cm，
   零动作会先仰到 -0.167 rad 再被 PD 拉回，停在非零平衡点。
   必须做固定点迭代把脚心挪到质心正下方。
5. **`push_robot` 与 `reset_base` 的 z 随机化会污染开环测试** ——
   冒烟测试的 quiet 模式要把它们弹掉。

### 下一步
开训 `Mjlab-Velocity-Flat-Pikachu-V03`，`num_envs=2048`，`save_interval=500`。
