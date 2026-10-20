"""Three real PPO trials demonstrating autoresearch keep/discard in isolation."""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

TASK = "Mjlab-Velocity-Flat-Pikachu-V033-E1"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--actor-init", type=Path, required=True)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[1]
  out = args.out.resolve()
  out.mkdir(parents=True, exist_ok=False)
  repo = out / "recipe_repo"
  repo.mkdir()

  def git(*command: str) -> str:
    return subprocess.check_output(["git", *command], cwd=repo, text=True).strip()

  git("init", "-b", "autoresearch/v033-e1-demo")
  git("config", "user.name", "Pikachu Experiment")
  git("config", "user.email", "experiment@localhost")
  (repo / "program.md").write_text(
    "# Bounded RL autoresearch demo\n\n"
    "Only recipe.json learning_rate may change. Three trials, 64 envs x 30 x 24 steps.\n"
    "All trials use seed 42 and the same actor-only initialization. Fresh critic/optimizer.\n"
    "Frozen nominal validation: seed 123, 8 envs x 6s, stand/forward/turn.\n"
    "Score = sum(mean_axis(abs velocity error / [0.4,0.4,0.3]))) + 100*falls.\n"
    "Lower wins; keep if improvement > 0.001. Never edit formal E1 configs.\n"
    "Fixed transition budget, unlike upstream's fixed 5-minute wall-clock budget.\n"
    "This demonstrates orchestration; one seed and tiny budget cannot select a final recipe.\n"
  )
  env = dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
  prefix = ["uv", "run", "--no-sync", "python"]
  best_score, best_commit = float("inf"), ""
  results = []

  def run(command: list[str], log: Path) -> None:
    with (out / "commands.jsonl").open("a") as stream:
      stream.write(
        json.dumps({"argv": command, "cwd": str(root), "time": time.time()}) + "\n"
      )
    with log.open("w") as stream:
      subprocess.run(
        command,
        cwd=root,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=300,
      )

  for index, learning_rate in enumerate((3e-4, 1e-4, 3e-3)):
    recipe = {"learning_rate": learning_rate}
    (repo / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    git("add", "program.md", "recipe.json")
    git("commit", "-m", f"Trial {index}: learning rate {learning_rate}")
    commit = git("rev-parse", "HEAD")
    git("branch", f"trial-{index}", commit)
    trial = out / f"trial_{index}"
    began = time.monotonic()
    result = {
      "commit": commit,
      "learning_rate": learning_rate,
      "status": "crash",
      "score": None,
    }
    try:
      run(
        prefix
        + [
          "scripts/train_v033_e1.py",
          "--task",
          TASK,
          "--out",
          str(trial),
          "--num-envs",
          "64",
          "--iterations",
          "30",
          "--learning-rate",
          str(learning_rate),
          "--actor-init",
          str(args.actor_init.resolve()),
        ],
        out / f"trial_{index}_train.log",
      )
      score = 0.0
      for clip in ("stand", "forward", "turn"):
        run(
          prefix
          + [
            "scripts/review_v033_policy.py",
            "--task",
            TASK,
            "--checkpoint",
            str(trial / "model_29.pt"),
            "--out",
            str(trial / "validation"),
            "--seed",
            "123",
            "--num-envs",
            "8",
            "--seconds",
            "6",
            "--clip",
            clip,
          ],
          out / f"trial_{index}_{clip}.log",
        )
        metrics = json.loads(
          (trial / "validation" / f"{clip}_metrics.json").read_text()
        )
        score += (
          sum(
            error / scale
            for error, scale in zip(metrics["mae"], (0.4, 0.4, 0.3), strict=True)
          )
          / 3
        )
        score += 100 * metrics["falls_total"]
      kept = score < best_score - 0.001
      result.update(score=score, status="keep" if kept else "discard")
      result["peak_torch_reserved_mib"] = json.loads(
        (trial / "progress.json").read_text()
      )["peak_torch_reserved_mib"]
      if kept:
        best_score, best_commit = score, commit
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
      result["error"] = repr(exc)
    result["seconds"] = time.monotonic() - began
    results.append(result)
    # Only our two-file isolated repository is reset. Trial branches retain all
    # candidates; logs/results/checkpoints are outside it and never deleted.
    if best_commit:
      git("reset", "--hard", best_commit)
    (out / "results.json").write_text(
      json.dumps(
        {
          "trials": results,
          "best_commit": best_commit,
          "best_score": best_score if best_commit else None,
        },
        indent=2,
      )
      + "\n"
    )
    with (out / "results.tsv").open("w") as stream:
      writer = csv.DictWriter(
        stream,
        fieldnames=("commit", "learning_rate", "score", "status", "seconds"),
        delimiter="\t",
        extrasaction="ignore",
      )
      writer.writeheader()
      writer.writerows(results)
    print(json.dumps(result), flush=True)
  if not best_commit:
    raise RuntimeError("All demo trials failed; inspect preserved logs.")
  print(f"Demo complete; retained recipe commit: {best_commit}", flush=True)


if __name__ == "__main__":
  main()
