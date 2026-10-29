# V033 双足 / 四足实验审核入口

**2026-10-20 E1正式长训：** 用户指定直立E1、四足E1各4096环境×20000轮。有效参数、冻结源码、后台队列、自动评估/视频和进度入口见 [E1正式训练记录](e1_full_20261020.md)。autoresearch三次真实短测及Git保留/回退体验见 [autoresearch_demo.md](autoresearch_demo.md)。长训最终效果以自动生成报告和最终checkpoint视频为准。

本轮已完成：双足原始 E0 基线与独立四足 Crawl，各 **3000 轮、1024 环境、seed=42**，每项 73,728,000 条转移样本。原配置默认 50000 轮，本轮仅为首轮审核结果。分支：`feat/pikachu-v033-crawl-20261019`。

日志根目录现为 **`logs/rsl_rl/pikahcu_v033/`**，`pikahcu` 按用户指定拼写。2026-10-20 完成迁移，旧的 `logs/v033_20261019/` 已移除。模型和视频未重新训练或改写。

**当前直立 E1（2026-10-20）：** 独立任务 `Mjlab-Velocity-Flat-Pikachu-V033-E1`，日志 `pikahcu_v033/biped_e1`，保留18维输出与512/256/128网络。按用户范围加入8个臂关节软约束，开启定向臂腿mesh接触与10mm提前惩罚，强化静止姿态/速度/双支撑奖励。`BIPED_KEYFRAME`现在为左roll=-53°、右roll=+53°，其余臂关节0；共享此keyframe的当前E0也会采用新手臂初态，历史E0权重和报告保持原样。新参数、碰撞调研、静止验收与回放见 **[biped_e1.md](biped_e1.md)**。

直立E1的最新短程候选为`2026-10-20_22-13-28_pilot-v2-robust-standing/model_499.pt`：512环境×500轮。标称8环境×10秒及随机化16环境×30秒均通过静止阈值；前进仍有侧向漂移和少量手臂越界，转向基本站住、未通过。4段候选录像、失败pilot和固定checkpoint回放命令均见上述文档，未标记为全向行走验收通过。

**当前四足设计（2026-10-20）：** 新增独立 `Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1`，实验名为 `pikahcu_v033/crawl_e1`，恢复完整18维动作及原网络。双侧 arm_roll、arm_yaw 相对初态超出±5°时新增平方软惩罚，权重-1；不裁剪动作、不新增机械限位。初态为右roll=-90°、左roll=+90°，双yaw=+90°、双elbow=0°；yaw软窗口随初态变为[85°,95°]。elbow保留策略输出，零动作时目标为0°，不再由移除输出保证肘不动。

**最新接触与查看修复：** 已删除四足手部球形碰撞体，启用原elbow_link STL mesh接触；保留用户注释的ankle STL，当前每脚5条capsule碰撞。双足reset不再额外抬高1–5cm，四足速度debug arrows已启用并使用水平heading方向。参数调整位置、GUI路径及检查结果见 **[接触/站高/debug说明](contact_setup_20261020.md)**。

完整参数、惩罚公式、最新小批量checkpoint和回放见 **[crawl_e1.md](crawl_e1.md)**。历史E1的64环境×5轮smoke来自yaw=0°和球形接触，尚无当前mesh接触下的正式训练效果。roll/yaw软窗口与原机械限位相交，软惩罚不保证永不越界。

当前初态/动作默认偏置已与历史3000轮训练不同；下列旧四足checkpoint、视频及报告指标仍对应原姿态，不能当成E1结果。恢复18维后形状一致也不代表旧策略适配新姿态。旧姿态见 `config_audit/crawl_calibration.json`；两次姿态调整分别见 `crawl_pose_20261020.json`、[crawl_pose_yaw0_elbow_hold_20261020.json](config_audit/crawl_pose_yaw0_elbow_hold_20261020.json)，后者保留16动作的历史检查，已由E1设计替代。现有Viser进程不会热加载源码，需要重启。

## 先看这些文件

| 内容 | 入口 |
|---|---|
| 直立E1手臂范围、臂腿避碰、静止优化与验证 | [biped_e1.md](biped_e1.md) |
| 完整训练参数、奖励权重与实测结果 | [training_report.md](training_report.md) |
| Crawl E1新设计、短测参数与最新回放 | [crawl_e1.md](crawl_e1.md) |
| 最新碰撞、双足站高调整位置、四足debug箭头 | [contact_setup_20261020.md](contact_setup_20261020.md) |
| 双足最终视频，64秒 / 8个片段 | [biped_e0/review_all.mp4](../../logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/evaluation/review/videos/review_all.mp4) |
| 四足最终视频，64秒 / 8个片段 | [crawl/review_all.mp4](../../logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/evaluation/review/videos/review_all.mp4) |
| 工作记录、失败 pilot 与限制 | [WORKLOG.md](WORKLOG.md) |
| 状态、最终 checkpoint 与 SHA256 | [status.json](../../logs/rsl_rl/pikachu_v033/_audit/20261019/metadata/status.json) |
| 训练曲线 / 曲线数据 | [PNG](../../logs/rsl_rl/pikachu_v033/_audit/20261019/reports/figures/training_curves.png) · [CSV](../../logs/rsl_rl/pikachu_v033/_audit/20261019/reports/figures/training_curves.csv) |
| 交付视频与模型核验 | [artifact_verification.json](../../logs/rsl_rl/pikachu_v033/_audit/20261019/reports/artifact_verification.json) |
| 逐文件迁移映射与迁移前后 SHA256 | [migration_manifest.json](../../logs/rsl_rl/pikachu_v033/_audit/20261019/reports/migration_manifest.json) |
| 本次整理、文档链接与回放启动核验 | [organization_verification.json](../../logs/rsl_rl/pikachu_v033/_audit/20261019/reports/organization_verification.json) |

## 原训练 checkpoint 与当前姿态的 play 指令

原训练的两条 checkpoint 命令均明确指定 **`model_2999.pt`**，不依赖自动选择最新目录。E1的最新小批量模型命令见 [crawl_e1.md](crawl_e1.md)。先进入仓库：

```bash
cd /home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab
```

**双足 E0：**

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-E0 \
  --checkpoint-file logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/training/main/checkpoints/model_2999.pt \
  --num-envs 1 --device cuda:0 --viewer viser
```

**四足 Crawl：**

以下为历史18动作模型的命令。当前恢复18动作，但初态/动作偏置已变；严格复现原效果需使用训练时源码快照。此命令不能用于评价E1，当前姿态可用后面的零动作命令查看。

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-Crawl \
  --checkpoint-file logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/training/main/checkpoints/model_2999.pt \
  --num-envs 1 --device cuda:0 --viewer viser
```

**查看当前E1四足初态（零动作）：**

```bash
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1 \
  --agent zero --balance False --freeze-joints False \
  --num-envs 1 --device cuda:0 --viewer viser
```

零动作由真实位置执行器保持初态，受力时关节可以有偏差。当前mesh接触下的物理检查见 [physics_check.json](../../logs/rsl_rl/pikachu_v033/crawl_e1/diagnostic_archive_20261022/evaluation/mesh_contact_20261020/metrics/physics_check.json)，双足reset/四足mesh与Viser控件核验见 [setup_check.json](../../logs/rsl_rl/pikachu_v033/crawl_e1/diagnostic_archive_20261022/evaluation/mesh_contact_20261020/metrics/setup_check.json)。之前yaw90阶段的[几何/碰撞审计](config_audit/crawl_pose_yaw90_20261020.json)及16动作阶段的[姿态检查结果](../../logs/rsl_rl/pikachu_v033/crawl/diagnostic_archive_20261022/evaluation/pose_yaw0_elbow_hold_20261020/metrics/result.json)、[截图](../../logs/rsl_rl/pikachu_v033/crawl/diagnostic_archive_20261022/evaluation/pose_yaw0_elbow_hold_20261020/images/initial_side.png)、[视频](../../logs/rsl_rl/pikachu_v033/crawl/diagnostic_archive_20261022/evaluation/pose_yaw0_elbow_hold_20261020/videos/zero_action_5s.mp4)作为历史保留。

启动后打开终端打印的 Viser 地址，通常为 `http://localhost:8080`；两条依次运行，按 `Ctrl+C` 退出当前回放。远程运行时可转发对应端口。命令使用普通 Viser 策略回放，不启用冻结关节或外部平衡辅助；桌面原生窗口可去掉 `MUJOCO_GL=egl` 并把 `--viewer viser` 改成 `--viewer native`。

2026-10-20 姿态/动作维度修改前，原两条 checkpoint 命令各做30秒启动检查。之后16动作阶段的Crawl零动作命令另做20秒Viser启动检查，日志在 `crawl/diagnostics/pose_yaw0_elbow_hold_20261020/play_zero.log`；E1本轮验证了物理、训练、checkpoint加载和离屏录像链路。原加载检查日志保留在 `_audit/20261019/play_biped_20261020.log` 与 `play_crawl_20261020.log`；这些均不代替人工交互审核。

`play` 使用任务的 play 配置及其交互/采样命令，不等同于报告的固定命令标称评估。报告评估关闭 startup 域随机化和推力；需要复核固定命令时使用下节的评估脚本。交付视频是确定性策略回放，持续摆臂不能归因于录像时的随机探索。

| 子任务 | checkpoint | SHA256 |
|---|---|---|
| 双足 E0 | `logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped/training/main/checkpoints/model_2999.pt` | `3f720ea0f2935fe28fc3aacab57aba67b35ec384f3315c9be0ba5ba2b513eded` |
| 四足 Crawl | `logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/training/main/checkpoints/model_2999.pt` | `69aacc9d70e699edbe327a2a52e61cfacc32718207b9d8d1978f73ad4fd4cf41` |

## 目录组织

```text
logs/rsl_rl/pikahcu_v033/
├── biped_e0/
│   ├── 2026-10-19_23-01-42_review-biped/
│   │   ├── model_*.pt, *.onnx, params/, git/, events.out.tfevents.*
│   │   ├── train.log, eval.log, video.log
│   │   └── review/  # 最终视频、量化评估 JSON/CSV、视频合辑
│   └── diagnostics/  # smoke、链路检查与中间检查点评估
├── crawl/
│   ├── 2026-10-20_00-20-57_review-crawl/
│   │   ├── model_*.pt, *.onnx, params/, git/, events.out.tfevents.*
│   │   ├── train.log, eval.log, video.log
│   │   └── review/
│   └── diagnostics/  # smoke、pilot、pilot_revised、零动作与中间评估
├── crawl_e1/
│   └── diagnostics/  # E1物理/边界检查、5轮smoke模型与链路录像
└── _audit/20261019/
    ├── status.json, commands.jsonl, migration_manifest.json
    ├── artifact_verification.json, training_curves.png/csv
    ├── source_snapshot.tar.gz, snapshot-files.txt, pip-freeze.txt
    ├── workspace_snapshot_20261020.tar.gz
    ├── historical_metadata/  # 路径更新前 JSON 原件
    └── *.log  # 公共检查、历史流水线及报告日志
```

每项 `review/` 内有 `stand/forward/backward/turn × side/front` 共8段原始视频，均720p、50fps、8秒；`review_all.mp4` 保留全部帧、添加命令和视角标签，共64秒。`*_metrics.json/csv` 为独立的32环境×12秒量化评估，`*_side/front.json/csv` 对应单环境录像。自动 reset 与表现不佳的片段均保留。

`diagnostics/` 是预检与中间结果，不能代替最终结果。原 `logs/rsl_rl/pikachu_v033_e0/` 中其他既存实验未迁移，本轮以本页明确列出的目录为准。

本目录中的文档按用途整理：

| 位置 | 内容 |
|---|---|
| `README.md` | 审核入口、固定 checkpoint 回放和目录说明 |
| `training_report.md` | 自动生成的参数与评估报告 |
| `crawl_e1.md` | E1设计、完整动作、软窗口、短测结果与专属回放命令 |
| `WORKLOG.md` | 按时间追加的工作记录；早期旧路径保留为历史记录 |
| [config_audit/asset_audit.json](config_audit/asset_audit.json) | 质量、限位、执行器参数、动作缩放与碰撞体 |
| [config_audit/crawl_calibration.json](config_audit/crawl_calibration.json) | 四足 IK 与闭环标定 |
| [config_audit/e0_sensor_order_audit.json](config_audit/e0_sensor_order_audit.json) | E0 左右传感器配对证据 |
| `provenance/` | 实验前差异、工作区状态、基线提交与快照校验值 |

原始 `commands.jsonl`、训练日志、保存的 `params/*.yaml`、模型和训练时源码快照保持原文，因此可能出现旧路径或旧 experiment_name；通过迁移清单查找现位置。更新路径的 JSON 原件在 `historical_metadata/`，JSON 内的评估数值不变。训练源码快照与整理后的工作区快照分开保存，分别校验。

## 固定命令复核与报告维护

以下为原配置下对历史最终四足模型的评估示例（严格复现需恢复训练时初态/配置；当前E1单独评估）：

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/review_v033_policy.py \
  --task Mjlab-Velocity-Flat-Pikachu-V033-Crawl \
  --checkpoint logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl/training/main/checkpoints/model_2999.pt \
  --clip forward --seconds 12 --num-envs 32 \
  --out logs/rsl_rl/pikachu_v033/crawl/manual_review_20261020
```

可选 `--clip stand|forward|backward|turn`；双足换任务名与对应 checkpoint。录像加 `--video --num-envs 1 --seconds 8 --view side`（正面为 `front`），它会保留 reset 并同时输出 JSON/CSV。

查看两项正式训练曲线，仅选择本轮运行目录以避免混入 pilot：

```bash
uv run --no-sync tensorboard \
  --logdir_spec biped_e0:logs/rsl_rl/pikachu_v033/biped_e0/2026-10-19_23-01-42_review-biped,crawl:logs/rsl_rl/pikachu_v033/crawl/2026-10-20_00-20-57_review-crawl \
  --host 127.0.0.1
```

根据当前交付数据重新生成报告和曲线：

```bash
uv run --no-sync python scripts/report_v033_experiments.py
sha256sum -c doc/pikachu_v033_experiments_20261019/provenance/source_snapshot.sha256
sha256sum -c doc/pikachu_v033_experiments_20261019/provenance/workspace_snapshot_20261020.sha256
```

后续直接运行 `uv run train <任务名> ...` 时默认写入 `logs/rsl_rl/pikachu_v033/biped_e0`、`crawl/` 或 `crawl_e1/` 下的新时间戳目录。原两任务流水线脚本对已有 `status.json` 拒绝覆盖；E1短测单独运行并保存在diagnostics中。

## 人工审核关注点

- 双足：速度跟踪已学到，但双臂持续摆动；双臂8个关节参与策略输出，行走姿态奖励刻意放宽，没有独立摆臂幅度/速度验收。应补充手臂轨迹指标及约束对照，不能将此动作质量自动判为合格。
- 双足原始配置：足部接触右/左、高度扫描左/右，存在奖励配对问题；镜像反号与非对称关节限位不一致，应按真实机械依据核对。原基线未中途修改。
- 四足：前进目标0.15m/s，实测约0.143；三点及以上支撑仅47.9%。转向目标0.15rad/s，实测约0.0783。尚不能按严格三点支撑慢爬或准确全向控制验收。
- 四类命令各32环境×12秒，两任务均0跌倒；这是单种子标称评估，不代表鲁棒性或实机表现。碰撞与电机模型仍有简化，滑移指标为支撑site水平速度代理。

相关回归34通过/2跳过；新增部分通过独立检查。全仓仍有103项既存类型诊断，未记为全仓通过。现有源码保持可审查的未提交状态，未合并或上传；模型、视频和日志不纳入 Git。
