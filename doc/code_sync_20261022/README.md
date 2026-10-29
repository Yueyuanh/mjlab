# 4090 服务器同步

服务器：`finnox-4090@10.20.15.181`。
远程修改范围严格限制在 `/home/finnox-4090/RL/Pikachu/mjlab`。
在本地 mjlab 目录执行下列命令，SSH 按需提示输入密码，脚本不保存密码。

```bash
bash scripts/sync_4090.sh code --dry-run
bash scripts/sync_4090.sh code
```

同步 src、scripts、tests、typings、项目依赖声明和锁文件、V033 文档，
包含 V033 的 MJCF、URDF、STL 及共享训练框架依赖。按内容校验差异。
覆盖的旧文件备份在服务器 `.sync/backups/<UTC时间>-<进程号>/code/`。
不执行删除，不同步 Git、虚拟环境、训练日志、权重、部署导出和历史源码快照。
现有服务器训练进程保持运行，不自动安装依赖或启动训练。

两台机器各自在本机保存训练过程产物。训练完成后，仅按明确的最终模型路径传输，
不扫描并同步 `model_*.pt`，不把最大编号当作已完成训练的证据。
SHA-256 应来自已确认完成的训练记录，并在传输前后校验。

```bash
# 本地 -> 服务器，保留相对于 mjlab 的路径
bash scripts/sync_4090.sh push-final <最终模型相对路径> <SHA256>
# 服务器 -> 本地
bash scripts/sync_4090.sh pull-final <最终模型相对路径> <SHA256>
```

允许的模型目录为 `logs/rsl_rl/pikachu_v033/` 和 `pre_train/`；
模型命令每次只传一个明确文件，覆盖时也保留旧文件备份。
所有命令均从本地工作站运行。没有后台同步任务。

本次选择已有完成记录的两个正式最终模型（各自完成 5000 次迭代），
来源均为对应 `training/<run>/metadata/complete.json`，不包含 pilot、诊断或中间模型：

| 任务 | 模型相对路径 | SHA-256 |
| --- | --- | --- |
| Biped E1 | `logs/rsl_rl/pikachu_v033/biped_e1/gait_20261021/training/final_low_noise_512x5000/checkpoints/model_4999.pt` | `5dabf93c7244ee9fdc1a50d44fa0a7e06ba334f86277b013e23d41b6e65a492d` |
| Crawl E2 | `logs/rsl_rl/pikachu_v033/crawl_e2/compact_20261021/training/final_256x5000/checkpoints/model_4999.pt` | `bf922ae7ebfb167a20276240732d9ee126111c2ac6d0e47197403312df61b1b4` |

同步操作记录保存在本地 `.sync/20261022/`，完成后副本置于服务器同路径。

## 本次执行结果

- 首次同步新增或更新 109 个文件，文件总量约 30.6 MB；内容校验复查零差异。
- 首次覆盖备份：服务器 `.sync/backups/20261022T081519Z-113420/code/`。
- 服务器 V033 的 biped_e1、crawl、crawl_command、crawl_e1、crawl_e2 测试：
  `47 passed in 2.31s`，详见 `.sync/20261022/validation.log`。
- 上表两个最终模型已上传，传输前后 SHA-256 校验通过；下载命令也已验证。
- V033 远程模型目录仅包含上表两个 `.pt`，未传输中间权重。
- 服务器原有 V031-T6 训练进程 PID 1746253 仍在运行，未重启或停止。
- 未修改服务器 mjlab 目录以外的文件，未更改依赖环境、Git 分支或提交。
