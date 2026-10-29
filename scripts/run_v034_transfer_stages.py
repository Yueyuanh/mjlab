"""Sequential 500-update training, paired evaluation and standard-view videos."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(data, indent=2) + "\n")
  temporary.replace(path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--experiment", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--name", default="confirmed_mass_forward")
  parser.add_argument("--target", type=int, default=5000)
  args = parser.parse_args()
  if args.target <= 0 or args.target % 500:
    parser.error("target must be a positive multiple of 500")
  root = args.experiment.resolve()
  queue_file = root / f"metadata/{args.name}_queue.json"
  if queue_file.exists():
    raise FileExistsError(queue_file)
  checkpoint = args.checkpoint.resolve()
  state: dict[str, Any] = dict(
    status="starting",
    pid=os.getpid(),
    task=args.task,
    target=args.target,
    completed=0,
    milestones=[],
    initialization=str(checkpoint),
    started_at=time.time(),
    physical_authority="User-confirmed edited URDF mass, COM and inertia",
  )
  frozen = root / "provenance" / args.name
  frozen.mkdir(parents=True, exist_ok=False)
  hashes = {}
  paths = [
    ROOT / "scripts/train_v033_e1.py",
    ROOT / "scripts/review_v033_policy.py",
    Path(__file__).resolve(),
  ]
  paths += list(
    (ROOT / "src/mjlab/tasks/velocity/config/pikachu_v034_biped_transfer").glob("*.py")
  )
  paths += list(
    (ROOT / "src/mjlab/asset_zoo/robots/pikachu/xmls/pikachu_v034/mjcf").glob("*.xml")
  )
  for path in paths:
    relative = path.relative_to(ROOT)
    destination = frozen / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
  write_json(frozen / "sha256.json", hashes)
  environment = dict(
    os.environ,
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    MUJOCO_GL="egl",
  )

  def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
      subprocess.run(
        ["uv", "run", "python", *map(str, command)],
        cwd=ROOT,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
      )

  try:
    for count in range(500, args.target + 1, 500):
      # Refuse to mix physical assets or worker code within one experiment.
      for relative, digest in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
          raise RuntimeError(f"Source changed during queue: {relative}")
      name = f"{args.name}_{count:04d}"
      run_dir = root / "training" / name
      state.update(status="training", current_run=name)
      write_json(queue_file, state)
      command = [
        "scripts/train_v033_e1.py",
        "--task",
        args.task,
        "--out",
        run_dir,
        "--num-envs",
        "128",
        "--iterations",
        count,
        "--save-interval",
        "500",
        "--actor-init",
        checkpoint,
      ]
      if count == 500:
        command += [
          "--load-critic",
          "--reset-exploration",
          "--exploration-std",
          "0.20",
          "--learning-rate",
          "0.0003",
        ]
      else:
        command += ["--resume"]
      for bucket in ("standing", "forward", "backward", "lateral", "heading", "world"):
        command += ["--command-bucket", f"{bucket}={1 if bucket == 'forward' else 0}"]
      run(command, root / f"logs/training/{name}.log")
      complete = json.loads((run_dir / "complete.json").read_text())
      checkpoint = Path(complete["checkpoint"])
      if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != complete["sha256"]:
        raise RuntimeError("Completed checkpoint SHA mismatch")
      state.update(status="evaluating", completed=count)
      write_json(queue_file, state)
      case = root / "evaluation" / name
      base = [
        "scripts/review_v033_policy.py",
        "--task",
        args.task,
        "--checkpoint",
        checkpoint,
        "--clip",
        "forward",
      ]
      run(
        [*base, "--out", case / "metrics", "--seconds", "12", "--num-envs", "8"],
        root / f"logs/evaluation/{name}_metrics.log",
      )
      metrics = json.loads((case / "metrics/forward_metrics.json").read_text())
      for view in ("three_quarter", "front"):
        run(
          [
            *base,
            "--out",
            case / "videos",
            "--seconds",
            "6",
            "--num-envs",
            "1",
            "--video",
            "--view",
            view,
          ],
          root / f"logs/evaluation/{name}_{view}.log",
        )
      import imageio.v2 as imageio
      from PIL import Image

      images = case / "images"
      images.mkdir(exist_ok=True)
      videos = []
      for video in sorted((case / "videos").glob("*.mp4")):
        reader = imageio.get_reader(video)
        sheet = Image.new("RGB", (960, 360))
        for index, frame in enumerate((25, 100, 175, 250)):
          sample = Image.fromarray(reader.get_data(frame)).resize((480, 180))
          sheet.paste(sample, ((index % 2) * 480, (index // 2) * 180))
        reader.close()
        sheet.save(images / (video.stem + ".jpg"))
        videos.append(
          dict(path=str(video), sha256=hashlib.sha256(video.read_bytes()).hexdigest())
        )
      write_json(
        case / "metrics/artifacts.json",
        dict(checkpoint_sha256=complete["sha256"], videos=videos),
      )
      run(
        [root / "scripts/archive_run.py", name],
        root / f"logs/orchestration/{name}_archive.log",
      )
      checkpoint = run_dir / "checkpoints" / checkpoint.name
      complete["checkpoint"] = str(checkpoint)
      write_json(run_dir / "metadata/complete.json", complete)
      milestone = dict(
        completed=count,
        checkpoint=str(checkpoint),
        sha256=complete["sha256"],
        vx=metrics["actual_mean"][0],
        vx_mae=metrics["mae"][0],
        falls=metrics["falls_total"],
      )
      state["milestones"].append(milestone)
      write_json(queue_file, state)
      status = json.loads((root / "metadata/status.json").read_text())
      status.update(
        phase="training_evaluation_queue",
        training_active=False,
        latest_milestone=milestone,
        next_action="Next 500-update milestone",
      )
      write_json(root / "metadata/status.json", status)
    state.update(status="complete", finished_at=time.time())
    write_json(queue_file, state)
  except BaseException as exc:
    state.update(status="failed", error=repr(exc), updated_at=time.time())
    write_json(queue_file, state)
    raise


if __name__ == "__main__":
  main()
