"""Sequential frozen-source E1 training, nominal review videos and reports."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from train_v033_e1 import write_json

TASKS = {
  "biped_e1": "Mjlab-Velocity-Flat-Pikachu-V033-E1",
  "crawl_e1": "Mjlab-Velocity-Flat-Pikachu-V033-Crawl-E1",
}


def now() -> str:
  return datetime.now().astimezone().isoformat()


def report(audit: Path, state: dict[str, Any]) -> None:
  lines = [
    "# E1 正式训练运行报告",
    "",
    f"更新：{now()}；总状态：{state['status']}。",
    "",
    "每项 4096 环境 × 20000 PPO 轮 × 24 步 = 1,966,080,000 transitions。",
    "两项均随机初始化，seed=42；未加载历史 checkpoint。",
    "18 维动作，actor/critic 512/256/128 ELU；具体有效配置见各 params YAML。",
    "关闭只读 cfg.metrics 遥测，保留任务传感器、观测与奖励。",
    "一张 GPU 顺序执行；最终结果以评估 JSON 为准，不预先判定成功。",
    "",
  ]
  for name, experiment in state["experiments"].items():
    run = Path(experiment["run"])
    lines += [f"## {name}", "", f"状态：{experiment['status']}；目录：`{run}`。", ""]
    progress = run / "progress.json"
    if progress.exists():
      value = json.loads(progress.read_text())
      lines += [
        f"已完成 {value['completed_iterations']}/20000 轮；"
        f"最近单轮 {value['iteration_seconds']:.3f}s；"
        f"PyTorch 峰值保留显存 {value['peak_torch_reserved_mib']:.0f} MiB。",
        "",
      ]
    for result in sorted((run / "review").glob("*_metrics.json")):
      data = json.loads(result.read_text())
      lines += [
        f"- {result.stem}: command={data['command']}, "
        f"actual={data['actual_mean']}, MAE={data['mae']}, "
        f"falls={data['falls_total']}, survival={data['survival_fraction']}",
      ]
    for video in sorted((run / "review").glob("*.mp4")):
      lines += [f"- [{video.name}]({video})"]
    lines += [
      "",
      "```bash",
      f"uv run --no-sync play {TASKS[name]} --checkpoint-file {run}/model_19999.pt "
      "--num-envs 1 --viewer viser",
      "```",
      "",
    ]
    if experiment.get("errors"):
      lines += [f"错误（保留日志）：{experiment['errors']}", ""]
  lines += [
    "评估为固定命令、无外推/冻结/扶持的 nominal 测试；不等于实机验证。",
    "直立重点审核静止踱步、手臂越界、转向和碰撞；四足重点审核前臂接触与滑移。",
  ]
  target = audit / "training_report.md"
  temporary = target.with_suffix(".tmp")
  temporary.write_text("\n".join(lines) + "\n")
  temporary.replace(target)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--audit", type=Path, required=True)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--root", type=Path, required=True)
  args = parser.parse_args()
  audit, source, root = args.audit.resolve(), args.source.resolve(), args.root.resolve()
  if (audit / "status.json").exists():
    raise FileExistsError("Queue status exists; use a new audit directory.")
  tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  state: dict[str, Any] = {
    "status": "running",
    "started": now(),
    "pid": os.getpid(),
    "source": str(source),
    "experiments": {
      name: {
        "task": task,
        "status": "queued",
        "run": str(root / "logs/rsl_rl/pikahcu_v033" / name / f"{tag}_full-4096x20000"),
        "errors": [],
      }
      for name, task in TASKS.items()
    },
  }
  env = dict(
    os.environ,
    MUJOCO_GL="egl",
    OMP_NUM_THREADS="4",
    PYTHONUNBUFFERED="1",
    PYTHONPATH=str(source / "src"),
  )

  def persist() -> None:
    write_json(audit / "status.json", state)
    report(audit, state)

  def run(command: list[str], log: Path) -> None:
    with (audit / "commands.jsonl").open("a") as stream:
      stream.write(
        json.dumps({"time": now(), "argv": command, "cwd": str(source)}) + "\n"
      )
    with log.open("a") as stream:
      child = subprocess.Popen(
        command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT
      )
      state["child_pid"] = child.pid
      persist()
      while child.poll() is None:
        time.sleep(10)
        persist()
      state.pop("child_pid", None)
      if child.returncode:
        raise RuntimeError(f"exit={child.returncode}; log={log}")

  persist()
  for name, experiment in state["experiments"].items():
    task, run_dir = experiment["task"], Path(experiment["run"])
    experiment.update(status="training", started=now())
    persist()
    try:
      run(
        [
          "uv",
          "run",
          "--no-sync",
          "python",
          "scripts/train_v033_e1.py",
          "--task",
          task,
          "--num-envs",
          "4096",
          "--iterations",
          "20000",
          "--out",
          str(run_dir),
        ],
        audit / f"{name}_train.log",
      )
      checkpoint = run_dir / "model_19999.pt"
      completion = json.loads((run_dir / "complete.json").read_text())
      if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != completion["sha256"]:
        raise RuntimeError("Final checkpoint hash mismatch")
      experiment.update(
        status="reviewing", checkpoint=str(checkpoint), sha256=completion["sha256"]
      )
      persist()
    except Exception as exc:
      experiment["errors"].append(repr(exc))
      experiment.update(status="training_failed", finished=now())
      persist()
      continue
    base = [
      "uv",
      "run",
      "--no-sync",
      "python",
      "scripts/review_v033_policy.py",
      "--task",
      task,
      "--checkpoint",
      str(checkpoint),
      "--out",
      str(run_dir / "review"),
    ]
    (run_dir / "review").mkdir(exist_ok=True)
    for clip in ("stand", "forward", "backward", "turn"):
      commands = [base + ["--clip", clip, "--seconds", "12", "--num-envs", "32"]]
      commands += [
        base
        + [
          "--clip",
          clip,
          "--seconds",
          "8",
          "--num-envs",
          "1",
          "--video",
          "--view",
          view,
        ]
        for view in ("side", "front")
      ]
      for command in commands:
        try:
          run(command, audit / f"{name}_review.log")
        except Exception as exc:
          experiment["errors"].append(repr(exc))
    video_manifest = {}
    for clip in ("stand", "forward", "backward", "turn"):
      try:
        metrics = json.loads((run_dir / "review" / f"{clip}_metrics.json").read_text())
        if metrics["checkpoint"] != str(checkpoint) or metrics["task"] != task:
          raise ValueError("Evaluation checkpoint/task mismatch")
      except (OSError, ValueError, KeyError) as exc:
        experiment["errors"].append(f"{clip} metrics: {exc!r}")
      for view in ("side", "front"):
        video = run_dir / "review" / f"{clip}_{view}.mp4"
        try:
          metadata = json.loads(
            subprocess.check_output(
              [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,width,height:format=duration",
                "-of",
                "json",
                str(video),
              ],
              text=True,
            )
          )
          if float(metadata["format"]["duration"]) < 7.9:
            raise ValueError("Video shorter than requested eight seconds")
          video_manifest[video.name] = {
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "probe": metadata,
          }
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
          experiment["errors"].append(f"{video.name}: {exc!r}")
    write_json(run_dir / "review" / "video_manifest.json", video_manifest)
    experiment.update(
      status="complete_with_review_errors" if experiment["errors"] else "complete",
      finished=now(),
    )
    persist()
  state.update(
    status="complete"
    if all(e["status"] == "complete" for e in state["experiments"].values())
    else "finished_with_errors",
    finished=now(),
  )
  persist()


if __name__ == "__main__":
  main()
