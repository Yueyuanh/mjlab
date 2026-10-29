"""Run an audited training worker and categorize files only after it exits."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--experiment", type=Path, required=True)
  parser.add_argument("--run", required=True)
  args, worker = parser.parse_known_args()
  if worker and worker[0] == "--":
    worker = worker[1:]
  experiment = args.experiment.resolve()
  root = Path(__file__).resolve().parents[1]
  for key, value in {
    "WARP_CACHE_PATH": root / ".sync/cache/warp",
    "XDG_CACHE_HOME": root / ".sync/cache",
    "CUDA_CACHE_PATH": root / ".sync/cache/cuda",
    "TMPDIR": root / ".sync/tmp",
  }.items():
    value.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(value)
  out = experiment / "training" / args.run
  log = experiment / "logs/training" / f"{args.run}.log"
  log.parent.mkdir(parents=True, exist_ok=True)
  if out.exists():
    raise FileExistsError(out)
  with log.open("x") as stream:
    code = subprocess.call(
      [sys.executable, "scripts/train_v033_e1.py", "--out", str(out), *worker],
      stdout=stream,
      stderr=subprocess.STDOUT,
    )
  if out.exists():
    for child in list(out.iterdir()):
      if child.name in (
        "checkpoints",
        "configs",
        "tensorboard",
        "metadata",
        "provenance",
      ):
        continue
      category = (
        "checkpoints"
        if child.suffix == ".pt"
        else "metadata"
        if child.suffix == ".json"
        else "configs"
        if child.name == "params"
        else "tensorboard"
        if child.name.startswith("events.out.tfevents.")
        else "provenance"
      )
      target = out / category
      target.mkdir(exist_ok=True)
      child.rename(target / child.name)
    complete = out / "metadata/complete.json"
    if complete.exists():
      value = json.loads(complete.read_text())
      value["checkpoint"] = str(out / "checkpoints" / Path(value["checkpoint"]).name)
      complete.write_text(json.dumps(value, indent=2) + "\n")
  status = {"exit_code": code, "run": str(out), "pid": os.getpid()}
  (experiment / "metadata").mkdir(exist_ok=True)
  (experiment / "metadata" / f"{args.run}_exit.json").write_text(
    json.dumps(status, indent=2) + "\n"
  )
  raise SystemExit(code)


if __name__ == "__main__":
  main()
