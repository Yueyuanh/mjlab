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

全部 **11 个 mp4**，`ffprobe` 逐个校验为 `1280×720 / 50 fps / 300 帧`，合计 15 MB。

> ⚠️ **下表是旧规格（自由相机、6 秒）的产物**，正按上面的新规格（跟踪相机、
> 5 秒、文件名带取景档 `{档位}__{取景}__{tag}.mp4`）重录，录完会整表替换。

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
  --out docs/pikachu_v03/videos --tag r05-best \
  --clips stand,fwd,fwd_fast,back,lat,turn --views side --seconds 5
# 站姿开度必须单独跑一趟 front（side 档看不见左右间距）
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python scripts/record_v03_video.py \
  --checkpoint logs/rsl_rl/pikachu_v03_velocity/<run>/model_29999.pt \
  --out docs/pikachu_v03/videos --tag r05-best \
  --clips stand,lat --views front --seconds 5
```

一个 `(档位, 取景)` 组合建一个 env —— 相机是在 `OffscreenRenderer.__init__`
里一次定死的，换取景只能重建 env（事后改 `env._offline_renderer._cam` 是捅私有属性）。

每档打印 `step / render / write` 分项耗时。正常基准
**~12 + 6 + 2 ms/帧**；若某一档突然几百秒，先看分项 ——
实测踩过一次「把 300 帧 720P（~830 MB）攒在内存里再 `mimsave`」，
在 swap 已满的机器上编码从 6 s 涨到 **517 s**（已改为流式写入）。

取景数字怎么来的：`geom_aabb` 是 geom **局部系**的包围盒，旋转后取角点会放大
（hip 的 mesh 转了 35°，量出来宽 0.458 m，真实只有 0.362 m）。要准就得拿
`mesh_vert` 的真实顶点变换到世界系，再手工投影 —— 全程只用
`mjv_updateCamera`，不建 GL 上下文、不步进物理，内存几十 MB。
