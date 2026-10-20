# 实验来源与代码快照

- `base_commit.txt`：创建实验分支时的 Git 基线提交。
- `preexisting-status.txt`：实验前工作区状态。
- `preexisting.patch`：实验前已存在的跟踪文件差异；不是本次实验的独立补丁，未跟踪文件由源码快照补充。
- `source_snapshot.sha256`：本轮训练交付时保存的源码快照校验值。该快照保持不变，包含日志迁移前的脚本路径和任务 experiment_name。
- `workspace_snapshot_20261020.sha256`：2026-10-20 日志整理后的代码与文档快照校验值。它反映当前复核脚本和目录布局，不应冒充训练开始时的源码。

快照、依赖版本、迁移清单和原始日志统一放在
`logs/rsl_rl/pikahcu_v033/_audit/20261019/`。两份校验文件均从 mjlab 仓库根目录运行
`sha256sum -c <校验文件路径>`。

代码仍处于可审查的未提交状态，既存修改未被覆盖。当前实验分支名及固定 checkpoint
回放指令见上级 [README](../README.md)。
