# 开源训练仓库 logging 调研

调研日期：2026-10-29。以下结论基于各仓库固定 commit 的源码，而不是只参考 README 或当前网页；
链接也固定到相同 commit，便于之后复核。

## 对比结果

| 仓库 | 固定版本 | 观察到的做法 | 对 mjlab 的取舍 |
| --- | --- | --- | --- |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/0ca25c56eec6fe7860fdcba44f8848f19693d6f6) | `0ca25c56` | run 保存解析后的 env/agent 配置和 `run.json`；恢复时按 task/algorithm 等兼容条件筛选 | 采用有效配置与单一 run manifest，不采用扁平 run 列表 |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl/tree/11563319134f987f70fb034410496fe3cc7327a9) | `11563319` | checkpoint 包含 iteration/optimizer 等恢复状态；rank 0 负责共享写入；TensorBoard 与 W&B 可并存；logger 可保存 Git diff | 采用可恢复 checkpoint、单写入者、本地日志和代码身份；W&B 只作外部索引 |
| [legged_gym](https://github.com/leggedrobotics/legged_gym/tree/8fa29acc6fd1910c3d9659eef6310bdd301cde0a) | `8fa29acc` | `logs/<experiment>/<datetime>_<run_name>`，run 内直接交给 runner 管理；resume 从实验目录寻找 run/checkpoint | 保留 experiment/run 两级和 run 内聚；用四阶段目录解决 datetime run 过多后的扁平列表问题 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl/tree/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb) | `fe8d8a03` | run 名通常编码 env、experiment、seed、time；本地 TensorBoard 可与 W&B 同步；部分流程训练后直接评估并录视频 | 采用可区分的 run 身份和训练后自动评估；日期只保留在 run ID，不重复塞进 experiment 名 |
| [Habitat Lab](https://github.com/facebookresearch/habitat-lab/tree/0fb6f43ffe806a8088a171b036336c093bcf604e) | `0fb6f43f` | checkpoint、TensorBoard、video、log 使用明确路径；checkpoint 保存 config；W&B resume 有明确 ID/模式 | 采用 artifact 边界、配置随 checkpoint 可追溯和严格 resume 身份；不把外部 tracker 当作唯一证据 |

## 共同模式

五个仓库在目录名称上没有统一答案，但有几项稳定共识：

1. **run 必须有稳定身份**：时间戳本身不够，至少还要能区分任务、配置/实验和 seed 或重复批次。
2. **配置必须是实际生效值**：恢复或比较依赖 run 启动时解析后的配置，不依赖后来变化的默认代码。
3. **checkpoint 必须可恢复且可定位**：模型权重之外，还要保存 iteration、optimizer 等训练状态；正式
   候选不能靠“最新文件”猜测。
4. **本地与外部 logger 分工**：TensorBoard/W&B 适合看训练过程，关键配置、checkpoint 和最终评估仍需
   保留本地副本；外部 run ID 用于关联而不是替代。
5. **训练后评估应自动化**：训练曲线不能回答最终行为是否达标，固定 checkpoint 的独立评估和视频应
   成为 full run 的结束步骤。
6. **代码状态属于实验身份**：commit、dirty 状态和必要 diff 能避免“同名配置、实际代码不同”的不可
   复现问题。

## 本规范为何采用阶段优先

`legged_gym` 的简单时间 run 很适合少量实验，但 run 多以后，pilot、失败重跑和正式 full 会混在同一
列表。完全按 artifact 类型拆目录又会破坏 run 原子性：复制一个 run 时必须跨多个目录拼装配置、模型、
指标和视频。

因此 mjlab 采用：

```text
robot/task/experiment/stage/run/artifacts
```

这一顺序同时回答五个日常问题：哪个机器人、哪个任务、验证什么假设、处于哪一级、是哪次重复。打开
experiment 后首先看到四个固定阶段，需要审阅正式结果时可以直接进入 `4-full/`。run 内仍保留开源
仓库验证过的配置快照、恢复状态、本地 telemetry、代码身份和训练后评估。

## 未直接照搬的做法

- 不采用纯日期时间 run 名：可排序，但看不出配置序号和重复批次。
- 不采用实验根按 `checkpoints/`、`videos/`、`reports/` 横向拆分：会让一个 run 的证据散落。
- 不依赖 W&B artifact 作为唯一归档：离线、权限变化或项目迁移后仍需本地复现。
- 不让工具自动选择“最新 checkpoint”：需要 `config/run.json` 的完成状态、路径和 SHA-256。
- 不把训练回报当最终评估：full 结束后必须运行固定速度数据录制和独立报告。
