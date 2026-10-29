# 2026-10-15 本机 / 246 服务器代码统一与 Mirror005 模型下放

## 目标与范围

- 本机仓库：`/home/finnox-4090/RL/Pikachu/mjlab`
- 服务器：`finnox@10.20.13.246`
- 服务器仓库：`/home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab`
- 模型目录：`logs/rsl_rl/pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42`
- 推理检查点：`model_49999.pt`；同时传输该批次全部 207 个文件（约 1.14 GB）。

统一的是源码、任务配置、机器人资产、测试和文档的工作目录内容。双方原有 Git
HEAD、暂存区、远程地址、虚拟环境、日志与机器专用 `.codex/.agents` 配置保留。
未执行 Git push，也未将训练权重或新生成的遥测数据提交到 Git。

## 同步前状态与备份

- 本机 HEAD：`6d546de8bf6e2849f9465ed513040e34e89db650`。
- 服务器 HEAD：`f1bcc86ff3b7c0d5a1a69ce2732fe8e2baf292b6`。
- 共同祖先：`76692ef92822bb638dfd3fa4166b9cd0b80ea3c2`。
- 本机备份：仓库同级 `code_sync_backups/20261015-mirror005/local-before/`。
- 服务器备份：仓库同级 `code_sync_backups/20261015-mirror005/remote-before/`。

每份备份包含 `worktree.tar.gz`、`history.bundle`、`manifest.json`、
`working.patch`、`index.patch`。包含已跟踪文件和非忽略的未跟踪文件；不包含
虚拟环境、日志和机器专用隐藏配置。备份目录也保存了独立的 integration 预览、
来源选择清单和校验清单。原有模型和旧部署资料未删除。

## 合并取舍

| 部分 | 合并结果 |
|---|---|
| V031 Mirror005 / Mirror010、E1–E4、BetaBounded | 保留本机任务名、超参数、观测和动作语义 |
| V031 T6 / Clip / Sym / SymClip / YawHuber | 保留服务器任务入口及各自 runner 配置 |
| 镜像实现 | Mirror005 继续用 `pikachu_v031_data_augmentation`；旧 Sym/SymClip 使用兼容入口 `mirror_augmentation`，原实现存于 `symmetry_legacy.py` |
| 足底遥测 / CoP / viewer 叠加 / dashboard | 合入服务器已提交及未提交改进 |
| metrics 录制入口 | 同时支持 `--mode nominal/dr/train` 和旧 `--nominal/--with-push/--obs-noise`；保留动作裁剪及 E4 课程固定逻辑 |
| V03-R07 / V03-T6 | 两个入口都保留；T6 的软扭矩上限按 3 N·m 配置 |
| V03 轻/重资产 | 按用户选择，同名基础 XML 保留 2.381 kg，重模型使用独立 `weight6.3.xml` |
| V031 source MJCF | 合入服务器 left_hip_yaw_link 质量对称性修正，与双方已有 train MJCF / URDF 保持一致 |
| 报告 | 本机 Mirror005/Beta 报告和服务器 T6/足底报告均保留 |

## 验证

- 模型目录 `rsync --checksum --dry-run` 无差异。
- 镜像、实验任务、足底遥测、速度奖励、任务注册、CSV dashboard、reward panel
  定向测试：69 passed、2 skipped。
- 模型及源码 SHA256、服务器 GPU 回放结果见 `verification.json`。
- 服务器同一组测试：69 passed、2 skipped。
- RTX 4060 Laptop GPU 加载 `model_49999.pt` 成功；标称静止、前进各 3 秒
  均跑满 150 步且无终止。前进命令 0.40 m/s，实测约 0.42 m/s，净位移
  1.18 m，逐帧 xy 误差均值 0.047 m/s；静止误差约 0.002 m/s。
- GPU 验证 CSV 与索引也已取回本机，见 `metrics/`。该 3 秒检查用于确认模型迁移
  可运行，不替代原训练评估报告的长时间、完整命令组测试。

## 服务器使用

```bash
cd /home/finnox/Pikachu/PikachuRobot/pikachu_playground/mjlab
export PATH="$HOME/.local/bin:$PATH"

uv run --no-sync list-envs

uv run --no-sync play Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005 \
  --checkpoint-file \
  logs/rsl_rl/pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42/model_49999.pt \
  --viewer velocity-joystick
```

离线固定命令评估：

```bash
uv run --no-sync python scripts/record_v03_metrics.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-E0-Mirror005 \
  --checkpoint \
  logs/rsl_rl/pikachu_v031_e0_mirror005/2026-10-12_00-11-23_mirror005-s42/model_49999.pt \
  --out doc/code_sync_20261015/metrics \
  --tag mirror005-server-smoke --clips stand,fwd --seconds 3 \
  --device cuda:0 --mode nominal
```

后续应基于这份统一工作目录继续开发。双方 Git 历史尚未生成正式合并提交；若要
提交，应先按仓库规范完成 `make check`，并仅暂存源代码/配置/正式报告，避免把
训练权重、临时 CSV、曲线或旧 mocap 中间数据混进提交。
