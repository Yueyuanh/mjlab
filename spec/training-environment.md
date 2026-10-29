# 训练环境规范

## 依赖与运行入口

- 仓库命令必须通过 `uv run` 或 Makefile 入口执行，依赖以 `uv.lock` 为准。
- 启动前必须记录 Python、mjlab commit、工作树状态/补丁哈希、MuJoCo/MuJoCo Warp、Torch、CUDA、
  GPU 型号与驱动。Docker 训练还要记录 image digest。
- 不要求工作树绝对干净，但必须保存能重建训练源码的 commit 加 diff；训练期间不得继续修改该
  run 所使用的源码。更改后启动新 run。
- CLI 参数必须从当前版本 `--help` 核对。布尔值、点路径和连字符以实际解析结果为准。

## 启动前检查

- 确认 Task ID、有效 env/agent 配置、seed、环境数、迭代数、保存间隔和输出目录。
- 检查 GPU 占用、显存余量、磁盘余量、进程冲突和预计 checkpoint 体积。
- 长训练前必须完成对应任务的 sanity、针对性测试和 smoke。
- 首轮训练应启用 NaN/finite 检查；发现非有限状态时保留诊断快照，不继续盲跑。
- seed 必须显式记录。固定 seed 不能保证 GPU 训练逐 bit 可复现，报告不得作此承诺。

## 运行与监控

- 训练 stdout/stderr 保存到当前 run 的 `tensorboard/train.log`；编排器日志若不能写入单个 run，保存
  在实验根 README 明确链接的位置。不得重新建立实验级 `logs/training/` artifact 树，并须保留退出码。
- 应当监控迭代进度、吞吐、loss、学习率、显存、系统内存、NaN 和 checkpoint 写入状态。
- 监控脚本不得移动、删除或重写活跃 run 文件；评估活跃 checkpoint 时先确认写入已经原子完成。
- 多 GPU/多节点训练必须保证每个进程 seed 不重复，并记录 world size、rank 到 GPU 的映射。
- W&B 是远程镜像和观察面板，不是唯一事实来源；断网时本地训练和元数据仍应完整。

## 机器同步

- 代码、环境、Git 状态和活跃日志按机器隔离。同步脚本不得删除远端独有文件。
- 不镜像正在写入的完整训练目录。最终模型只按 `config/run.json` 中的明确相对路径传输。
- 传输前后都必须校验 SHA-256；源、目标、时间和校验结果写入同步或交付记录。
- 远程修改、备份和报告必须留在用户指定的项目目录内，不在 home 或系统目录散落文件。
- 密码、SSH key 和 token 不进入仓库；认证由机器环境提供。

本仓库 4090 的具体地址和同步命令仍以根 `AGENTS.md` 为准，本文件只规定通用行为。
