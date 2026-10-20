"""Freeze effective source and assets, including dirty dependencies, with hashes."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[1]
  out = args.out.resolve()
  out.mkdir(parents=True, exist_ok=True)
  source = out / "source"
  source.mkdir(exist_ok=False)
  for name in ("src", "scripts", "tests", "typings", "docs"):
    shutil.copytree(
      root / name,
      source / name,
      ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "_build"),
    )
  for name in (
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "README.md",
    "LICENSE",
    ".gitignore",
  ):
    shutil.copy2(root / name, source / name)
  manifest = {
    str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(source.rglob("*"))
    if path.is_file()
  }
  (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
    archive.add(source, arcname="source")
  (out / "source.sha256").write_text(
    hashlib.sha256((out / "source.tar.gz").read_bytes()).hexdigest()
    + "  source.tar.gz\n"
  )
  for name, command in (
    ("git_head.txt", ["git", "rev-parse", "HEAD"]),
    ("git_branch.txt", ["git", "branch", "--show-current"]),
    ("git_status.txt", ["git", "status", "--short"]),
    ("workspace.patch", ["git", "diff", "--binary", "HEAD"]),
    ("uv_freeze.txt", ["uv", "pip", "freeze"]),
    ("nvidia-smi.txt", ["nvidia-smi"]),
  ):
    (out / name).write_bytes(subprocess.check_output(command, cwd=root))
  # Reuse the installed environment; PYTHONPATH in the queue selects this source.
  (source / ".venv").symlink_to(root / ".venv", target_is_directory=True)
  print(f"Frozen {len(manifest)} files: {source}")


if __name__ == "__main__":
  main()
