# Pikachu v03 录像

720P（1280×720 @ 50 fps，**5 秒 = 250 帧**）无头离线渲染，由
`scripts/record_v03_video.py` 生成。

> **命令是「构造 env 之前把 `twist.ranges` 收成一个点」固定的**，
> 并把所有 `rel_*_envs` 归零 —— 比事后改 `vel_command_b` 干净，
> 观测里的 command 项也天然自洽。
> ⚠️ 必须 `cfg.curriculum.pop("command_vel", None)`，否则课程项每步把
> ranges 覆写回默认值（见 [../04-worklog.md](../04-worklog.md)）。

## 相机：`AUTO` 不是「自动跟踪」，是「自由相机」

「看不清效果」的根因是**取景**，不是分辨率（一直就是 720P）。
`ViewerConfig` 默认 `origin_type=AUTO`，而 `offscreen_renderer.py::_get_camera_body_id`
把 `AUTO` 和 `WORLD` **都**返回 `-1`，于是相机被钉在世界原点上：
`lookat=(0,0,0)`、`distance=5.0`、`elevation=-45`。把默认站姿的 mesh 顶点
投影到 1280×720 上实测，机器人只占 **12% 画面高、2% 画面宽**；而且相机不动，
机器人以 0.4 m/s 走 6 s（2.4 m）就飘出画面中心。

改成 `ASSET_ROOT` 得到 **tracking 相机**，实测两件事：

| 实测 | 结果 | 含义 |
|---|---|---|
| 相机 `lookat` vs `data.subtree_com[root]` | **逐位相等** | 跟的是**整机质心**，不是 `xpos`；走路时画面平滑不抖 |
| 机器人平移 2 m + 绕 z 转 90° 后的 `scn.camera[0].forward` | **一字未变** | **只跟位置、不跟朝向** → `turn` 档看到的是机器人原地转身，而不是整个世界在转 |

取景档位由投影占比定（`--views`）：

| 档位 | azimuth | elevation | distance | 占画面高 | 占宽 | 用途 |
|---|---|---|---|---|---|---|
| `side` | 90° | −15° | 1.1 m | 58% | 10% | 默认；机器人朝画面**右**走 |
| `front` | 0° | −15° | 0.9 m | 64% | 29% | **唯一能看清双脚横向间距的档** |
| `three_quarter` | 45° | −15° | 1.1 m | 57% | 19% | 折中 |

> ⚠️ **站姿横向开度在 `side` 档里根本看不见** —— 侧视看到的是脚的前后深度，
> 不是左右间距。r05 专门加了 `feet_lateral_distance` 奖励（见
> [../03-task-design.md](../03-task-design.md)），要验证它就必须看 `front` 档。

### ⚠️ 换成 tracking 之后，`viewer.entity_name` 也必须显式给

`AUTO` 在查 entity **之前**就 `return -1` 了，所以下面这段一直没被执行过；
一旦真的去查，`_get_camera_body_id` 只在场景**恰好一个** entity 时才肯自动检测：

```python
if entity_name is None:
    if len(entities) != 1:
        raise ValueError(f"Cannot auto-detect entity: scene has {len(entities)} entities ...")
```

flat 任务里 terrain 也是一个 entity（`Scene._add_terrain` 会
`self._entities["terrain"] = terrain`），所以 `scene.entities` 实际是
`{"robot", "terrain"}` —— **2 个**，必抛。必须写
`env_cfg.viewer.entity_name = "robot"`。

> 全仓**没有**别的任务设过这个字段（`g1`/`go1`/`pikachu*` 都只设 `body_name`
> 且用默认 `AUTO`）—— 也就是说这条路径在本仓库是第一次被走到。

## 文件

全部 **17 个 mp4**，`ffprobe` 逐个校验为 `1280×720 / 50 fps / 250 帧`（5 秒），
合计 49 MB。文件名是 `{档位}__{取景}__{tag}.mp4`。

### `r04-best`（7 个）—— 稳定行走的基线

检查点 `logs/rsl_rl/pikachu_v03_velocity/2026-10-01_14-48-28_v03-flat-r04/model_29999.pt`

| 文件 | 档位（命令） | 大小 |
|---|---|---|
| `stand__side__r04-best.mp4` | 站立（0, 0, 0） | 1265 KiB |
| `stand__front__r04-best.mp4` | 站立 —— **看站姿开度用这个** | 1480 KiB |
| `fwd__side__r04-best.mp4` | 前进 0.40 m/s | 4971 KiB |
| `back__side__r04-best.mp4` | 后退 0.30 m/s | 4690 KiB |
| `lat__side__r04-best.mp4` | 侧移 0.25 m/s | 4547 KiB |
| `lat__front__r04-best.mp4` | 侧移 | 4299 KiB |
| `turn__side__r04-best.mp4` | 转向 0.30 rad/s（⚠️ 分布外，见下） | 1977 KiB |

### `r05-15000`（8 个）—— 加了站姿奖励、**但没有门控**的那一轮

检查点 `logs/rsl_rl/pikachu_v03_velocity/2026-10-08_19-45-08_v03-flat-r05/model_15000.pt`

| 文件 | 档位 | 大小 |
|---|---|---|
| `stand__side__r05-15000.mp4` | 站立 | 1495 KiB |
| `stand__front__r05-15000.mp4` | 站立 —— 与 r04 的 front 并排看，开度差一眼可见 | 1785 KiB |
| `fwd__side__r05-15000.mp4` | 前进 0.40 m/s | 4855 KiB |
| `fwd_fast__side__r05-15000.mp4` | 前进 0.50 m/s | 1364 KiB |
| `back__side__r05-15000.mp4` | 后退 0.30 m/s | 4515 KiB |
| `lat__side__r05-15000.mp4` | 侧移 | 4439 KiB |
| `lat__front__r05-15000.mp4` | 侧移 | 4574 KiB |
| `turn__side__r05-15000.mp4` | 转向 | 1379 KiB |

> `fwd_fast` 只有 r05 有（上一轮中断的录像留下的，规格相同），r04 无对照。

### `r01-before`（2 个）—— 修奖励之前的失败形态

检查点 `logs/rsl_rl/pikachu_v03_velocity/2026-10-01_12-17-25_v03-flat-r01/model_500.pt`

| 文件 | 档位 | 大小 |
|---|---|---|
| `stand__side__r01-before.mp4` | 站立（对照） | 632 KiB |
| `fwd__side__r01-before.mp4` | 前进 0.40 m/s（对照） | 1327 KiB |

## 怎么看这三组

**r01 vs r04**：`r01-before` 是**修奖励之前的失败形态** —— 命令前进 0.40 m/s，
机器人原地不动（实测 0.001 m/s）。两组并排看，能一眼看出「站着不动」和
「真的在走」的区别。这比看奖励曲线可靠（`Episode_Reward/track_*` 会因为
Jensen 不等式骗人，见 [../reports/training-report.md](../reports/training-report.md) §3.4）。

**r04 vs r05（本轮重点）**：并排看 `stand__front__*` 两个文件。
r04 站立时双脚间距 **9.23 cm**，r05 是 **16.41 cm**（默认 keyframe 20.00 cm）
—— 奖励确实起作用了。但再看 `fwd__side__*`：r05 的**原地踏步**很明显，
脚抬得比 r04 勤，身体却不往前走。逐项数据在
[../04-worklog.md](../04-worklog.md)（`track_linear_velocity` 0.72→0.38，
而 `error_vel_yaw` 反而 0.77→0.39）。

**文件大小本身就是旁证**：r01 的 `fwd` 是 **1327 KiB**，r04 的 `fwd` 是
**4971 KiB** —— 机器人不动时画面几乎静止，H.264 压缩率自然高。注意这只能当
旁证不能当判据（`turn` 档 r01 反而更大，因为 r01 在乱晃而 r04 稳定）。

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
# 一组 = 一个模型；side 档录步态，front 档单独录（站姿开度只在 front 看得见）
for V in side front; do
  [ "$V" = side ] && CLIPS=stand,fwd,back,lat,turn || CLIPS=stand,lat
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python scripts/record_v03_video.py \
    --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt \
    --out docs/pikachu_v03/videos --tag r06-best \
    --clips "$CLIPS" --views "$V" --seconds 5
done
```

一组 7 个文件约 **50 秒**（实测 6.2–7.3 s/档，含建 env）。换模型就改
`--checkpoint` 和 `--tag`。

一个 `(档位, 取景)` 组合建一个 env —— 相机是在 `OffscreenRenderer.__init__`
里一次定死的，换取景只能重建 env（事后改 `env._offline_renderer._cam` 是捅私有属性）。

每档打印 `step / render / write` 分项耗时。正常基准
**~16 + 6 + 2 ms/帧**；若某一档突然几百秒，先看分项 ——
实测踩过一次「把 300 帧 720P（~830 MB）攒在内存里再 `mimsave`」，
在 swap 已满的机器上编码从 6 s 涨到 **517 s**（已改为流式写入）。

取景数字怎么来的：`geom_aabb` 是 geom **局部系**的包围盒，旋转后取角点会放大
（hip 的 mesh 转了 35°，量出来宽 0.458 m，真实只有 0.362 m）。要准就得拿
`mesh_vert` 的真实顶点变换到世界系，再手工投影 —— 全程只用
`mjv_updateCamera`，不建 GL 上下文、不步进物理，内存几十 MB。

**站姿开度的数值**不要从录像里目测 —— 用
`scripts/measure_v03_stance.py --clips stand`，它直接打 base 系 y 的间距均值
和左右 `hip_roll` 角。录像只用来**看**「原地踏步」这类时间上的形态。
