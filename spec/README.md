# mjlab 迭代与训练规范

状态：`v0.2-draft`
适用范围：Pikachu fork 中新增或继续进行的模型接入、任务开发、训练、评估与交付。
目标：让一次实验可解释、可复现、可比较、可追溯，并让 Agent 默认遵循同一套约定。

## 仓库经验简要总结

当前仓库已经形成了几项值得保留的习惯：

- 模型导入通过可重复执行的构建脚本完成，保留源文件 SHA-256、转换说明和几何审计结果；
- 任务由唯一 Task ID 绑定环境、回放环境、训练配置和 runner，并有任务谱系与状态；
- 训练前使用 `zero` / `random` agent 做 sanity check，再经过 smoke、pilot、full 分级放大；
- 正式 run 保留有效配置、TensorBoard、可恢复 checkpoint、代码身份和完成状态；
- 最终模型不靠“编号最大”猜测，而由选择记录、完成元数据和 checkpoint SHA-256 明确指定；
- 评估保存原始时序、汇总指标、视频、接触图和渲染元数据，结论区分 nominal 与 randomized；
- 报告会记录失败尝试和退化结果，而不只展示最终成功模型；
- 框架改动有小型 fixture 单测，具体机器人和任务另有针对性配置测试。

目前的问题也比较明确：规则散落在历史训练手册、`AGENTS.md`、`CLAUDE.md`、实验 README
和脚本注释中；部分历史命令已经过时；原生训练目录与归档目录并存但边界不够清楚；有些
实验缺少有效配置快照。此目录把稳定经验提升为规范，但不改写历史记录。

## 规范索引

| 主题 | 文档 | 何时阅读 |
| --- | --- | --- |
| Agent 默认行为与路由 | [agent-defaults.md](agent-defaults.md) | 所有 Agent 每次进入仓库 |
| 规范的维护与冲突处理 | [governance.md](governance.md) | 修改规范、遇到例外或冲突 |
| 模型导入与格式转换 | [model-import.md](model-import.md) | URDF/MJCF/mesh/电机参数接入 |
| 训练任务规范 | [training-task.md](training-task.md) | 新任务、奖励/观测/动作或训练变体 |
| 训练环境规范 | [training-environment.md](training-environment.md) | 配环境、同步机器、启动训练 |
| 实验、日志与模型保存 | [experiment-logging.md](experiment-logging.md) | 建实验目录、续训、清理、晋级模型 |
| 评估与报告 | [evaluation-report.md](evaluation-report.md) | 模型选择、视频、对比、交付报告 |
| 开源 logging 调研 | [research/open-source-logging.md](research/open-source-logging.md) | 复核目录设计依据与取舍 |
| 不同 Agent 的默认接入 | [agent-integration.md](agent-integration.md) | 接入新的 Agent/IDE |
| 实验入口模板 | [templates/experiment-readme.md](templates/experiment-readme.md) | 创建新实验目录 |
| run README 模板 | [templates/run-readme.md](templates/run-readme.md) | 创建阶段内的新 run |
| metrics README 模板 | [templates/metrics-readme.md](templates/metrics-readme.md) | 记录 full 自动评估数据 |
| 评估报告模板 | [templates/evaluation-report.md](templates/evaluation-report.md) | 汇总正式候选与基线 |

## 快速使用

1. 新工作先确定属于哪一类，并读取对应主题规范。
2. 在训练前创建实验入口 README，并按 `1-sanity/`、`2-smoke/`、`3-pilot/`、`4-full/` 建立
   `E<实验序号>-R<批次>-<YY-MM-DD>` run。
3. 使用仓库入口和 `uv run`，先做针对性测试与小规模验证，再启动长训练。
4. 训练结束后由 `config/run.json` 的完成状态、checkpoint SHA-256 和模型选择记录指定候选，不按
   文件名或修改时间猜测。
5. 用固定协议评估，保存机器可读数据，再写结论和交付状态。

历史训练手册和实验报告继续作为事实记录；新工作若与本目录冲突，以本目录为准，并在必要时给旧
文档增加 superseded 提示。历史实验不要求一次性迁移。
