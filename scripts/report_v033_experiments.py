"""Build the Chinese review report from actual checkpoints and evaluation JSON."""

import json

import yaml
from plot_v033_training import main as plot_training
from v033_artifacts import (
  AUDIT,
  ROOT,
  audit_status,
  experiment_runs,
  review_dir,
  training_dir,
)

OUT = ROOT / "doc/pikachu_v033_experiments_20261019"
LOGS = AUDIT


def main():
  state = json.loads(audit_status().read_text())
  plot_training()
  runs = experiment_runs()
  lines = [
    "# V033 双足基线与四足爬行训练审核报告",
    "",
    f"状态：{state['status']}；开始时间：{state['started']}；更新时间取自 status.json。",
    "",
    "分支：`feat/pikachu-v033-crawl-20261019`。原工作区有大量既存修改，已保存 provenance/preexisting.patch / provenance/preexisting-status.txt；没有覆盖或提交这些既存修改。",
    "",
    "## 配置检查",
    "",
    "E0：4.702736 kg，22 个关节，其中 18 驱动 / 4 被动四杆关节；50 Hz 策略 / 200 Hz 物理；左右镜像增强与 mirror loss=0.05。",
    "J4310 额定配置 12.5 N·m，J3507 3.0 N·m。此处为仿真扭矩限幅，未建模完整电机转速-扭矩曲线或热限制。",
    "原四足 QUAD_KEYFRAME 为双足占位，不能直接训练；新任务使用独立配置包、独立初态、前肢球形接触、四通道接触观测、世界竖直 yaw reset、水平 heading 速度观测/奖励和前俯姿态终止。",
    "E0 保留原训练配方；其仅脚底接触模式不模拟手臂/躯干触地和大部分自碰撞，必须在部署前补充更完整碰撞与硬件验证。",
    "源 XML 左右 ankle 限位同为[-0.79,1.05]、被动 knee 同为[-0.65,1.05]，而镜像符号表对应关节反号；非对称限位在反号映射下不保持不变。本轮保留来源，应核对真实机械限位。",
    "已确认 E0 接触顺序为右/左，高度扫描却为左/右；feet_swing_height 直接逐列组合两者，存在左右配对错误。foot_clearance 也将模型顺序的 site 速度与扫描高度组合。Crawl 使用显式一致的右后/左后/右前/左前顺序，无此扫描混配。E0 本轮作为原始配置基线保留，详见 config_audit/e0_sensor_order_audit.json。",
    "原 E0 默认 num_envs=1，不适合直接全量启动；本轮显式设置并行数。原任务默认 50000 轮，本轮为有限预算的首轮训练，不能声称 50k 收敛结果。",
    "双足双臂8关节参与策略动作，行走pose容差较宽，standing_posture_l2仅在近零命令生效；本轮未单独验收摆臂幅度、速度与能耗。最终录像为确定性策略，持续摆臂不能归因于录像时随机探索，也不能未经对照就判为合理自然摆臂。",
    "",
    "## 实际训练参数",
    "",
    "| 参数 | 双足 E0 | 四足 Crawl |",
    "|---|---|---|",
    f"| 迭代预算 | {state['iterations']} | {state['iterations']} |",
    f"| 并行环境 | {state['num_envs']} | {state['num_envs']} |",
    f"| 转移样本预算 | {state['iterations'] * state['num_envs'] * 24:,} | {state['iterations'] * state['num_envs'] * 24:,} |",
    "| 随机种子 | 42 | 42 |",
    "| 每轮步数 / batch | 24 / N×24 | 24 / N×24 |",
    "| actor / critic MLP | 512,256,128; ELU; obs normalization | 同左 |",
    "| actor / critic / action 维度 | 74 / 86 / 18 | 76 / 96 / 18（含2维节律相位） |",
    "| PPO epochs / minibatches | 5 / 4 | 5 / 4 |",
    "| 学习率 / 调度 | 0.001 / adaptive KL=0.01 | 同左 |",
    "| γ / GAE λ / clip / grad norm | 0.99 / 0.95 / 0.2 / 1.0 | 同左 |",
    "| 熵系数 / 初始动作 std | 0.01 / 1.0 | 0.005 / 0.5 |",
    "| 镜像增强 / mirror loss | 开 / 0.05 | 关，避免误用双足接触布局 |",
    "| checkpoint | 每 500 轮 + 结束 | 同左 |",
    "| 命令 x / y / yaw | 初始 [-.4,.5]/[-.35,.35]/[-.3,.3]；2000轮后[-.6,.8]/[-.5,.5]/[-.5,.5] | 前1000轮[.08,.20]/0/0；之后[-.1,.25]/[-.08,.08]/[-.2,.2] |",
    "| 静止样本 | 10% | 10% |",
    "| 爬行节律 | 无 | 周期1.6秒，每肢20%摆动、至多一肢摆动；抬高目标2cm |",
    "| 四足改进 | — | 跟踪std=.08m/s，Huber归一化尺度=.10m/s、权重-.75，节律接触奖励1.5 |",
    "| 扰动 | 原 E0 摩擦/COM/编码器/推力 | 摩擦/编码器/COM±5mm，无推力 |",
    "| actor 线/角速度均匀噪声 | ±0.5 m/s / ±0.2 rad/s | ±0.05 m/s / ±0.10 rad/s |",
    "| 动作硬裁剪 | 无；位置执行器及物理限位约束 | 同左 |",
    "",
    "单张 RTX 4060 Laptop 8 GB，顺序运行以避免争用。完整实际参数以每项 run 的 training/main/configs/env.yaml 与 agent.yaml（归档前为 params/） 为准。TensorBoard 本地记录，不向外部平台上传。",
    "",
    "## 实测结果",
    "",
    "以下评估固定 seed=42，关闭 startup 域随机化与推力，每命令 32 环境×12 秒；这是标称评估，不代表鲁棒性或多种子统计。跌倒自动 reset，视频保留 reset，MAE 含 reset 过渡。",
    "接触滑移列使用接触期间支撑 site 的世界水平速度作为代理；尤其球形前肢接触可能包含滚动，不等价于真实接触点的表面滑移。占空比顺序见配套JSON的contact_order。",
    "",
    "| 实验 / 命令 | 实测 vx/vy/wz | MAE vx/vy/wz | 无跌倒比例 | 总跌倒 | 接触占空比 | 接触时滑移速度m/s |",
    "|---|---|---|---|---|---|---|",
  ]
  reward_weights = {}
  for name, run in runs.items():
    configs = list(training_dir(run, "configs").glob("env.yaml"))
    if configs:
      # BaseLoader reads tags as plain data, without resolving Python objects.
      saved = yaml.load(configs[-1].read_text(), Loader=yaml.BaseLoader)
      reward_weights[name] = {
        key: value["weight"] for key, value in saved["rewards"].items()
      }
  reward_names = sorted({key for weights in reward_weights.values() for key in weights})
  reward_lines = [
    "## 实际奖励权重",
    "",
    "来自本轮已写出的有效 env.yaml；尚未启动的任务列留空。",
    "",
    "| 奖励项 | 双足 E0 | 四足 Crawl |",
    "|---|---|---|",
  ]
  for key in reward_names:
    reward_lines.append(
      f"| {key} | {reward_weights.get('biped', {}).get(key, '—')} | {reward_weights.get('crawl', {}).get(key, '—')} |"
    )
  reward_lines.append("")
  at = lines.index("## 实测结果")
  lines[at:at] = reward_lines
  statuses = []
  findings = []
  for name in ("biped", "crawl"):
    item = state["experiments"].get(name, {"status": "queued"})
    run = runs.get(name)
    current = list(training_dir(run, "checkpoints").glob("model_*.pt")) if run else []
    if current and "checkpoint" not in item:
      latest = max(current, key=lambda path: int(path.stem.split("_")[-1]))
      statuses += ["", f"{name} 已保存中间检查点：`{latest}`；这不是最终训练效果。"]
    metrics = sorted(review_dir(run, "metrics").glob("*_metrics.json")) if run else []
    for path in metrics:
      result = json.loads(path.read_text())

      def fmt(xs):
        return "/".join(f"{x:.3f}" for x in xs)

      lines.append(
        f"| {name}/{path.stem.removesuffix('_metrics')} | {fmt(result['actual_mean'])} | {fmt(result['mae'])} | {result['survival_fraction']:.1%} | {result['falls_total']} | {fmt(result['contacts_duty'])} | {fmt(result.get('contact_slip_speed_mean_m_s', []))} |"
      )
      if name == "crawl" and "three_or_more_support_fraction" in result:
        statuses += [
          f"{path.stem.removesuffix('_metrics')}：三点及以上支撑比例={result['three_or_more_support_fraction']:.1%}，完全腾空比例={result['zero_support_fraction']:.1%}。"
        ]
      clip = path.stem.removesuffix("_metrics")
      if result["falls_total"]:
        findings.append(
          f"{name}/{clip} 出现 {result['falls_total']} 次跌倒，不能仅按平均速度认定通过。"
        )
      if clip == "stand" and min(result["contacts_duty"]) < 0.9:
        findings.append(
          f"{name} 静止命令的接触检测占空比为 {fmt(result['contacts_duty'])}；需要审核原地踏步或接触抖动。"
        )
      if clip == "turn" and result["mae"][2] > abs(result["command"][2]) * 0.3:
        findings.append(
          f"{name} 原地转向不足：命令 {result['command'][2]:.3f}rad/s，实际均值 {result['actual_mean'][2]:.3f}rad/s，MAE {result['mae'][2]:.3f}rad/s。"
        )
      if (
        name == "crawl"
        and clip == "forward"
        and result.get("three_or_more_support_fraction", 1.0) < 0.8
      ):
        findings.append(
          f"四足前进虽能移动，三点及以上支撑仅 {result['three_or_more_support_fraction']:.1%}，尚不能按严格三点支撑慢爬验收。"
        )
    statuses += [
      "",
      f"{name} 状态：{item['status']}；checkpoint：`{item.get('checkpoint', '尚未完成')}`。",
      f"完成时间（含评估/录像）：{item.get('finished', '尚未完成')}；SHA256：`{item.get('sha256', '尚未完成')}`。",
      "",
    ]
  lines += statuses
  if findings:
    lines += ["## 审核关注点", "", *[f"- {finding}" for finding in findings], ""]
  lines += [
    "## 视频与复现",
    "",
    "流程完成后，视频使用最终实际 checkpoint 的确定性 actor，侧视和正视均为 1280×720、50fps、8 秒，包含静止、前进、后退、转向；没有筛除失败镜头。未完成时下面只列已实际产出的文件。",
  ]
  for name, run in runs.items():
    for path in sorted(review_dir(run, "videos").glob("*.mp4")):
      lines.append(f"- [{name}/{path.stem}]({path})")
  lines += [
    "",
    f"命令原文与状态在 `{LOGS}`；正式训练日志在各运行目录的 train.log，评估/录像日志为 eval.log、video.log。历史命令中的旧路径由 migration_manifest.json 映射至新位置。",
    "[目录与固定 checkpoint 回放指令](README.md)；原始训练参数 YAML 保留原文，experiment_name 反映迁移前名称。",
    "",
    "不能只凭奖励曲线判定学会四足：需要人工确认四肢确实交替支撑、没有拖肘滑行、没有持续穿模，结合速度误差与生存率审核。若退化或未收敛，本轮应标记不通过并据此继续调参。",
    "",
    f"[训练曲线]({LOGS / 'reports/figures/training_curves.png'}) · [曲线数据]({LOGS / 'reports/figures/training_curves.csv'}) · [工作记录](WORKLOG.md)",
    "两任务的奖励定义不同，曲线仅用于查看各自训练过程，不能按奖励数值直接排名。",
    "[机读参数审计](config_audit/asset_audit.json) 包含实际编译的质量、关节限位、执行器参数与动作缩放。",
    "",
    "验证：本次新增任务/脚本通过 ruff、ty、pyright；相关回归34通过/2跳过。全仓 make check -o format 剩余103项既有类型诊断，详见日志；没有把它记为全仓通过。",
  ]
  if state.get("error"):
    lines += ["", f"流程错误：`{state['error']}`"]
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "training_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
  main()
