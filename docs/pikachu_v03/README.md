# Pikachu v03 记录库

Pikachu **v03**（并联四杆膝，10 驱动关节 / 14 关节）在 mjlab 上的 velocity
行走任务：从 URDF/MJCF 资产接入，到静平衡站姿标定，到训练收敛的完整留痕。

> 与已有 `Pikachu-27dof` / V025（串联膝 14dof）是**两套独立资产**，不要混用。
> v03 的核心差异是膝：只有 `knee_up_joint` 被驱动，`knee_joint` /
> `knee_mid_joint` 是 `<connect>` 约束驱动的被动关节。

## 目录

| 文档 | 内容 |
|---|---|
| [01-model-conversion.md](01-model-conversion.md) | 源 MJCF → 训练 MJCF 的转换，含 4 个必踩的坑 |
| [02-stand-keyframe.md](02-stand-keyframe.md) | 站立 keyframe 标定：从「脚在髋下」到「质心在脚心上方」 |
| [03-task-design.md](03-task-design.md) | velocity 任务设计：动作缩放、奖励、命令空间、课程 |
| [04-worklog.md](04-worklog.md) | 工作日志（按时间倒序） |
| [reports/](reports/) | 训练报告 |
| [videos/](videos/) | 最优模型 720P 录像 |

## 快速上手

```bash
cd pikachu_playground/mjlab

# 重建训练 MJCF（改了源模型或胶囊参数后跑）
uv run python scripts/build_v03_train_xml.py

# 重标定静平衡站姿（改了质量/质心后跑）
uv run python scripts/probe_v03_both_legs.py     # 双腿独立解：脚掌俯仰 0° + 脚心 x=0
uv run python scripts/refine_v03_keyframe.py     # 固定点迭代：脚心挪到质心正下方

# 开训前的冒烟测试（分两个进程跑，本机内存不够同时建两个环境）
uv run python scripts/smoke_v03_velocity.py --quiet   # 零动作开环站立，必须过
uv run python scripts/smoke_v03_velocity.py --noisy   # 带 push 扰动，看难度

# 训练
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Pikachu-V03 \
  --env.scene.num-envs 2048
```

## 关键常量速查

| 量 | 值 | 出处 |
|---|---|---|
| 训练 MJCF | `src/mjlab/asset_zoo/robots/pikachu/xmls/mjcf_v03/pikachu_v03_base_series_train.xml` | `build_v03_train_xml.py` |
| 常量文件 | `src/mjlab/asset_zoo/robots/pikachu/pikachu_v03_constants.py` | — |
| 任务配置 | `src/mjlab/tasks/velocity/config/pikachu_v03/` | — |
| 站立高度 | 0.26728 m（静平衡微屈膝） | `refine_v03_keyframe.py` |
| 质量 | 2.3814 kg | 训练模型实测 |
| 质心 x | -0.02683 m | 静平衡点 |
| 动作维度 | 10 | 无手臂 |

## 一句话结论（持续更新）

- 2026-10-01：资产接入完成，静平衡站姿标定完成，零动作开环站立稳定
  （base_z 0.26728 → 0.26731，|v| → 0.00000，6 s 无终止）。可开训。
