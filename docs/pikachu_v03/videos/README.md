# Pikachu v03 录像

720P（1280×720 @ 50 fps，6 秒 = 300 帧）无头离线渲染，由
`scripts/record_v03_video.py` 生成。

> **命令是「构造 env 之前把 `twist.ranges` 收成一个点」固定的**，
> 并把所有 `rel_*_envs` 归零 —— 比事后改 `vel_command_b` 干净，
> 观测里的 command 项也天然自洽。
> ⚠️ 必须 `cfg.curriculum.pop("command_vel", None)`，否则课程项每步把
> ranges 覆写回默认值（见 [../04-worklog.md](../04-worklog.md)）。

## 文件

全部 **11 个 mp4**，`ffprobe` 逐个校验为 `1280×720 / 50 fps / 300 帧`，合计 15 MB。

| 文件 | 检查点 | 档位（命令） | 大小 |
|---|---|---|---|
| `stand__r04-best.mp4` | r04 @29999 | 站立（0, 0, 0） | 616 KiB |
| `fwd__r04-best.mp4` | r04 @29999 | 前进 0.40 m/s | 2768 KiB |
| `fwd_fast__r04-best.mp4` | r04 @29999 | 前进 0.50 m/s（命令上限） | 2859 KiB |
| `back__r04-best.mp4` | r04 @29999 | 后退 0.30 m/s | 2586 KiB |
| `lat__r04-best.mp4` | r04 @29999 | 侧移 0.25 m/s | 2465 KiB |
| `turn__r04-best.mp4` | r04 @29999 | 转向 0.30 rad/s（⚠️ 分布外，见下） | 968 KiB |
| `stand__r01-before.mp4` | r01 @500 | 站立（对照） | 697 KiB |
| `fwd__r01-before.mp4` | r01 @500 | 前进 0.40 m/s（对照） | 643 KiB |
| `back__r01-before.mp4` | r01 @500 | 后退 0.30 m/s（对照） | 367 KiB |
| `lat__r01-before.mp4` | r01 @500 | 侧移 0.25 m/s（对照） | 488 KiB |
| `turn__r01-before.mp4` | r01 @500 | 转向 0.30 rad/s（对照） | 834 KiB |

**最优检查点**：
`logs/rsl_rl/pikachu_v03_velocity/2026-10-01_14-48-28_v03-flat-r04/model_29999.pt`

> 仓库根的 `.gitignore` 有一条 `videos/`，会连目录一起排除。本目录是**显式
> 开的口子**（`.gitignore` 里 `!docs/pikachu_v03/videos/`），目的是让录像跟
> 文档一起进库、事后可 review；其它任何位置的 `videos/` 仍然被忽略。

## 怎么看这两组

`r01-before` 是**修奖励之前的失败形态**：命令前进 0.40 m/s，机器人原地不动
（实测 0.001 m/s）。`r04-best` 是修好之后并训满 30000 迭代的形态。
两组并排看，能一眼看出「站着不动」和「真的在走」的区别 ——
这比看奖励曲线可靠（`Episode_Reward/track_*` 会因为 Jensen 不等式骗人，
见 [../reports/training-report.md](../reports/training-report.md) §3.4）。

**文件大小本身就是旁证**：r01 的 `fwd` 是 **643 KiB**，r04 的 `fwd` 是 **2768 KiB**
—— 机器人不动时画面几乎静止，H.264 压缩率自然高。注意这只能当旁证不能当判据
（`turn` 档 r01 反而更大，因为 r01 在乱晃而 r04 稳定）。

## ⚠️ `turn` 档是**分布外**输入

`turn` 档给的是**恒定角速度** 0.30 rad/s，而训练配置是
`heading_command=True`：

```python
ang_vel_z = clip(heading_control_stiffness(0.5) × heading_error, ±0.3)
```

策略学的是「**转到目标朝向就停**」的**位置**任务，**从来没见过「一直转」**。
所以这个档里机器人几乎不转（实测 0.033），**不能读作「不会转向」**。

用分布内的测法（`scripts/eval_v03_turn.py`）重测：

| 场景 | r02@2500 收敛 | **r04@29999 收敛** |
|---|---|---|
| 原地转 90°（vx=0） | 0.06 | 0.18 |
| **边走边转 90°（vx=0.30）** | 0.09 | **0.98**（5.5 s 到位、残留 1.8°） |
| 原地转 180°（vx=0） | 0.03 | 0.63 |

**r04 会转向，但只在「边走边转」时** —— 那正是训练里 heading 环境的唯一形态。
原地转不会，因为训练分布里没有纯转向环境（r05 候选工作）。
详见 [../reports/training-report.md](../reports/training-report.md) §5.8。

## 重新生成

```bash
cd pikachu_playground/mjlab
# ⚠️ 必须等训练停下来 —— 训练期间起第二个仿真进程会换页到 7 s/帧（内存不够）
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python scripts/record_v03_video.py \
  --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt \
  --out docs/pikachu_v03/videos --tag r04-best \
  --clips stand,fwd,fwd_fast,back,lat,turn --seconds 6
```

每档打印 `step / render / write` 分项耗时。正常基准
**~12 + 6 + 2 ms/帧**；若某一档突然几百秒，先看分项 ——
实测踩过一次「把 300 帧 720P（~830 MB）攒在内存里再 `mimsave`」，
在 swap 已满的机器上编码从 6 s 涨到 **517 s**（已改为流式写入）。
