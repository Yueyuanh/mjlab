"""Script to inspect experiment runs: training commit, HEAD hash, mean reward.

Scans ``logs/rsl_rl`` and prints one row per run directory:

  | 实验名称 | commit | hash | Mean reward | 状态 |

* ``commit``  -- short hash of the git commit recorded at training start
  (parsed from the ``git/*.diff`` snapshot rsl_rl writes into each run dir).
* ``hash``    -- short hash of the current working tree HEAD.
* ``Mean reward`` -- last ``Train/mean_reward`` scalar from the tensorboard
  event file (``events.out.tfevents.*``).
* ``状态``    -- how the current code relates to the training commit:
    ✓  HEAD is exactly the training commit
    ≈  HEAD moved, but the working tree is content-identical to the training
       code (either ``git diff <commit>`` is empty, or applying the diff that
       was stored in the snapshot at training time reproduces the tree)
    ✗  the working tree no longer matches the training commit
    ?  cannot compare (commit missing / not a git repo)

Note: while a run is still training, the event file grows continuously, so
the reported reward is the latest value written so far.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import tyro
from prettytable import PrettyTable
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import mjlab

MEAN_REWARD_KEY = "Train/mean_reward"
GIT_COMMIT_MARKER = "--- git commit ---"
GIT_DIFF_MARKER = "--- git diff ---"
SHORT_HASH_LEN = 7


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
  """Run a git command inside the given working directory."""
  return subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(cwd))


def _repo_root() -> Path | None:
  """Return the git repository root containing the current working directory."""
  proc = _run_git(["rev-parse", "--show-toplevel"], Path.cwd())
  if proc.returncode != 0:
    return None
  return Path(proc.stdout.strip())


def _current_head_short(cwd: Path) -> str | None:
  """Return the short hash of the current HEAD, or None if not a git repo."""
  proc = _run_git(["rev-parse", "--short", "HEAD"], cwd)
  if proc.returncode != 0:
    return None
  return proc.stdout.strip()


def _read_train_snapshot(run_dir: Path) -> tuple[str, str] | None:
  """Parse (commit hash, stored diff) from the git snapshot rsl_rl wrote.

  The snapshot file format is::

      --- git commit ---
      <hash>

      --- git status ---
      ...

      --- git diff ---
      <diff of the working tree against HEAD at training start>

  Returns ``None`` if no usable snapshot is found.
  """
  git_dir = run_dir / "git"
  if not git_dir.is_dir():
    return None
  diff_files = sorted(git_dir.glob("*.diff"))
  if not diff_files:
    return None
  # Prefer the mjlab snapshot (the main repo of interest).
  mjlab_diff = git_dir / "mjlab.diff"
  if mjlab_diff.is_file():
    diff_files.insert(0, mjlab_diff)
  for diff_file in diff_files:
    try:
      text = diff_file.read_text(encoding="utf-8")
    except OSError:
      continue
    lines = text.splitlines()
    commit: str | None = None
    diff_start: int | None = None
    for i, line in enumerate(lines):
      stripped = line.strip()
      if commit is None and stripped == GIT_COMMIT_MARKER:
        for j in range(i + 1, len(lines)):
          if lines[j].strip():
            commit = lines[j].strip()
            break
      elif stripped == GIT_DIFF_MARKER:
        diff_start = i
        break
    if commit is not None:
      diff_text = ""
      if diff_start is not None:
        diff_text = "\n".join(lines[diff_start + 1 :]) + "\n"
      return (commit, diff_text)
  return None


def _commit_status(train_hash: str, stored_diff: str, repo: Path | None) -> str:
  """Compare the working tree against the training commit.

  Returns a one-character status symbol:
    ✓ HEAD is exactly the training commit.
    ≈ HEAD moved but the working tree is content-identical to the training
      code: either ``git diff <commit>`` is empty, or applying the diff that
      was stored in the snapshot at training time reproduces the tree.
    ✗ the working tree no longer matches the training commit.
    ? cannot compare (commit missing / not a git repo).
  """
  if repo is None:
    return "?"
  short = _run_git(["rev-parse", "--short", train_hash], repo)
  if short.returncode != 0:
    return "?"
  head_short = _current_head_short(repo)
  if head_short is not None and head_short == short.stdout.strip():
    return "✓"
  # HEAD differs: check whether the tree is still content-identical to the
  # training code.
  diff = _run_git(["diff", "--quiet", train_hash, "--"], repo)
  if diff.returncode == 0:
    return "≈"
  if stored_diff.strip():
    # The working tree should equal: training commit + stored diff.
    # Reverse-applying the stored diff must reproduce the commit cleanly.
    apply = subprocess.run(
      ["git", "apply", "--check", "--reverse", "-"],
      input=stored_diff,
      capture_output=True,
      text=True,
      cwd=str(repo),
    )
    if apply.returncode == 0:
      return "≈"
  return "✗"


def _mean_reward(run_dir: Path) -> str:
  """Return the latest ``Train/mean_reward`` from the tensorboard event file."""
  events = [p for p in run_dir.glob("events.out.tfevents.*")]
  if not events:
    return "N/A"
  # Pick the largest event file: it is the most complete one.
  events.sort(key=lambda p: p.stat().st_size, reverse=True)
  try:
    acc = EventAccumulator(str(events[0]))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if MEAN_REWARD_KEY in tags:
      key = MEAN_REWARD_KEY
    else:
      key = next((t for t in tags if "mean_reward" in t), None)
    if key is None:
      return "N/A"
    scalars = acc.Scalars(key)
    if not scalars:
      return "N/A"
    return f"{scalars[-1].value:.2f}"
  except Exception:
    return "N/A"


def list_experiments(
  log_root: str = "logs/rsl_rl",
  experiment: str | None = None,
  keyword: str | None = None,
) -> int:
  """List each training run with its commit, HEAD hash and mean reward.

  Args:
    log_root: Root directory containing experiment logs.
    experiment: Only list runs under this experiment directory name.
    keyword: Case-insensitive filter on the run directory name.
  """
  root = Path(log_root)
  repo = _repo_root()
  head_short = _current_head_short(Path.cwd()) if repo is not None else None

  rows: list[tuple[str, str, str, str, str]] = []
  if root.is_dir():
    exp_dirs = [p for p in root.iterdir() if p.is_dir()]
    if experiment:
      exp_dirs = [p for p in exp_dirs if p.name == experiment]
    for exp_dir in sorted(exp_dirs, key=lambda p: p.name):
      for run_dir in sorted(
        (p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: p.name
      ):
        if keyword and keyword.lower() not in run_dir.name.lower():
          continue
        # Only treat directories that look like a run (git snapshot or events).
        has_snapshot = (run_dir / "git").is_dir()
        has_events = any(run_dir.glob("events.out.tfevents.*"))
        if not has_snapshot and not has_events:
          continue
        snapshot = _read_train_snapshot(run_dir)
        status = (
          _commit_status(snapshot[0], snapshot[1], repo)
          if snapshot is not None
          else "?"
        )
        rows.append(
          (
            f"{exp_dir.name}/{run_dir.name}",
            snapshot[0][:SHORT_HASH_LEN] if snapshot is not None else "N/A",
            head_short or "N/A",
            _mean_reward(run_dir),
            status,
          )
        )

  rows.sort(key=lambda r: r[0], reverse=True)

  table = PrettyTable(["实验名称", "commit", "hash", "Mean reward", "状态"])
  table.title = "实验总览（最新在上）"
  table.align["实验名称"] = "l"
  for name, commit, head, reward, status in rows:
    table.add_row([name, commit, head, reward, status])
  print(table)
  print("状态: ✓ 代码一致   ≈ HEAD漂移但内容与训练时等价   ✗ 代码漂移   ? 无法比较")
  if not rows:
    print(f"[INFO] 未找到实验 run（log_root={log_root!r}）")
  return len(rows)


def main():
  return tyro.cli(list_experiments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
