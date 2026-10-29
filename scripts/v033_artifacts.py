"""Canonical paths for the V033 20261019 experiment review artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs/rsl_rl/pikachu_v033"
AUDIT = LOG_ROOT / "_audit/20261019"
TASK_DIRS = {"biped": LOG_ROOT / "biped_e0", "crawl": LOG_ROOT / "crawl"}


def experiment_runs() -> dict[str, Path]:
  """Prefer the recorded run, including when newer runs exist on disk."""
  state_file = audit_status()
  state = json.loads(state_file.read_text()) if state_file.exists() else {}
  runs = {}
  for name, directory in TASK_DIRS.items():
    recorded = state.get("experiments", {}).get(name, {}).get("run")
    if recorded:
      runs[name] = Path(recorded)
    else:
      candidates = sorted(directory.glob(f"*review-{name}"))
      if candidates:
        runs[name] = candidates[-1]
  return runs


def audit_status() -> Path:
  """Read historical native status or the categorized audit status."""
  archived = AUDIT / "metadata/status.json"
  return archived if archived.exists() else AUDIT / "status.json"


def training_dir(experiment: Path, category: str) -> Path:
  """Resolve the original workflow's single run after archival."""
  archived = experiment / "training/main" / category
  if (experiment / "training/main").exists():
    return archived
  return experiment / "params" if category == "configs" else experiment


def review_dir(experiment: Path, category: str) -> Path:
  """Resolve archived review data with native-run compatibility."""
  archived = experiment / "evaluation/review" / category
  return (
    archived if (experiment / "evaluation/review").exists() else experiment / "review"
  )
