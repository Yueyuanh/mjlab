"""Train and evaluate V034 baselines sequentially on one GPU, recording exact checkpoints."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def snapshot_sources(output):
  """Capture new/untracked V034 code that the runner's git diff would omit."""
  sources = list((ROOT / "scripts").glob("*v034*.py"))
  sources += list((ROOT / "src/mjlab/tasks/velocity/config/pikachu_v034").glob("*.py"))
  sources += [ROOT / "src/mjlab/asset_zoo/robots/pikachu/pikachu_v034_constants.py"]
  hashes = {}
  for source in sources:
    relative = source.relative_to(ROOT)
    destination = output / "source_snapshot" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
  assets = ROOT / "src/mjlab/asset_zoo/robots/pikachu/xmls/pikachu_v034"
  for source in sorted(assets.rglob("*")):
    if source.is_file():
      hashes[str(source.relative_to(ROOT))] = hashlib.sha256(
        source.read_bytes()
      ).hexdigest()
  (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


def run(command, logfile):
  print("Running:", " ".join(map(str, command)), flush=True)
  with logfile.open("w") as stream:
    subprocess.run(
      list(map(str, command)),
      cwd=ROOT,
      check=True,
      stdout=stream,
      stderr=subprocess.STDOUT,
      env={**os.environ, "MUJOCO_GL": "egl"},
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("both", "biped", "crawl"), default="both")
  parser.add_argument("--iterations", type=int, default=300)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--eval-envs", type=int, default=16)
  parser.add_argument("--seconds", type=float, default=10)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if (
    args.iterations < 1 or args.num_envs < 4 or args.eval_envs < 1 or args.seconds <= 0
  ):
    parser.error("iterations/eval-envs/seconds must be positive; num-envs must be >=4")
  label = "baseline_" + datetime.now().strftime("%Y%m%d_%H%M%S")
  output = (args.output or ROOT / "logs/rsl_rl/pikachu_v034" / label).resolve()
  output.mkdir(parents=True, exist_ok=True)
  snapshot_sources(output)
  manifest = {}
  for mode in ("biped", "crawl") if args.mode == "both" else (args.mode,):
    task = f"Mjlab-Velocity-Flat-Pikachu-V034-{mode.capitalize()}"
    train_cmd = [
      sys.executable,
      "-m",
      "mjlab.scripts.train",
      task,
      "--env.scene.num-envs",
      args.num_envs,
      "--agent.max-iterations",
      args.iterations,
      "--agent.save-interval",
      min(100, args.iterations),
      "--agent.seed",
      args.seed,
      "--agent.run-name",
      label,
      "--enable-nan-guard",
      "True",
    ]
    run(train_cmd, output / f"{mode}_train.log")
    folders = list((ROOT / "logs/rsl_rl/pikachu_v034" / mode).glob(f"*_{label}"))
    if len(folders) != 1:
      raise RuntimeError(f"Expected one exact run folder, found {folders}")
    checkpoint = max(
      folders[0].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])
    )
    manifest[mode] = {
      "task": task,
      "checkpoint": str(checkpoint),
      "train_command": list(map(str, train_cmd)),
    }
    (output / "runs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for evaluation in ("zero", "nominal", "randomized"):
      cmd = [
        sys.executable,
        "scripts/eval_v034.py",
        "--mode",
        mode,
        "--output",
        output / f"{mode}_{evaluation}.json",
        "--num-envs",
        args.eval_envs,
        "--seconds",
        args.seconds,
        "--seed",
        args.seed,
      ]
      if evaluation != "zero":
        cmd += ["--checkpoint", checkpoint]
      if evaluation == "randomized":
        cmd += ["--randomized"]
      run(cmd, output / f"{mode}_{evaluation}.log")
  print(f"Completed. Exact checkpoints and evaluations: {output}")


if __name__ == "__main__":
  main()
