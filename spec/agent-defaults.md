# Agent 默认指令

本文件是仓库内 Agent 的短入口。根目录的 Agent 原生配置负责自动引导到这里；不要把全部
主题规范复制进各 Agent 配置。

## 每次工作都必须遵守

1. 先确认任务范围和当前工作树，保留用户已有修改，不擅自清理、覆盖或移动实验产物。
2. 所有仓库 Python 命令使用 `uv run`；先用当前 CLI `--help` 核对参数，不照抄过时命令。
3. 训练与评估结论必须绑定 Task ID、有效配置、代码状态、checkpoint 路径与 SHA-256。
4. 不把“最新文件”“最高编号 checkpoint”或 W&B 页面状态当成训练完成和模型晋级的依据。
5. 训练遥测只用于观察优化过程；策略行为结论必须来自独立、固定协议的评估。
6. 长训练固定使用 `1-sanity/` -> `2-smoke/` -> `3-pilot/` -> `4-full/`；run 命名为
   `E<实验序号>-R<批次>-<YY-MM-DD>`，上一阶段失败时不得直接放大资源。
7. 修改代码后运行与改动风险相称的测试。提交前遵守根 `AGENTS.md` 的检查要求。
8. 一个 run 的配置、checkpoint、TensorBoard、指标、视频和报告必须在同一阶段 run 内；不移动活跃
   run 的 checkpoint、TensorBoard 或配置，归档、同步和清理都要有清单和哈希。
9. 不将密钥、密码、内网凭据写入脚本、日志、报告、配置快照或 Agent 指令。

## 按任务读取

| 工作内容 | 必读规范 |
| --- | --- |
| URDF/MJCF/mesh、关节、电机、碰撞体、格式转换 | `spec/model-import.md` |
| 新 Task ID、奖励、观测、动作、随机化、课程、训练变体 | `spec/training-task.md` |
| Python/CUDA/GPU、依赖、远程机、同步、启动训练 | `spec/training-environment.md` |
| 实验目录、日志、checkpoint、续训、清理、pre_train 晋级 | `spec/experiment-logging.md` |
| 评估脚本、指标、视频、基线对比、报告与交付 | `spec/evaluation-report.md` |
| 修改本规范或处理例外 | `spec/governance.md` |

同一任务涉及多个主题时读取所有相关文件。只做普通框架代码修改时，根 `AGENTS.md` 的开发
流程仍然适用；无需读取无关训练主题。
