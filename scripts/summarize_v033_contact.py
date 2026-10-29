"""Audit central stance separately from permitted touchdown/toe-off rolling."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--folder", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
meta = json.loads((args.folder / "rollouts.json").read_text())
raw = np.load(args.folder / "rollouts.npz")
derived = np.load(args.folder / "derived.npz")
start = round(meta["warmup_s"] / meta["dt"])
output = {}
for case in meta["cases"]:
  key = case["key"]
  output[key] = {}
  for side_index, side in enumerate(("right", "left")):
    index = next(
      i for i, n in enumerate(meta["contact_names"]) if n.endswith(side + "_ankle_link")
    )
    c = raw[key + "/contact"][..., index] > 0
    middle = np.zeros_like(c)
    for e in range(c.shape[1]):
      edges = np.diff(np.r_[False, c[:, e], False].astype(int))
      for lo, hi in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
        cut = max(1, int((hi - lo) * 0.2))
        middle[lo + cut : hi - cut, e] = True
    valid = middle[start:]
    tilt = derived[key + "/tilt"][start:, :, side_index][valid]
    gap = np.abs(derived[key + "/gap"][start:, :, side_index][valid]) * 1000
    output[key][side] = {
      "middle_tilt_mean_deg": float(tilt.mean()),
      "middle_tilt_p95_deg": float(np.percentile(tilt, 95)),
      "middle_gap_abs_mean_mm": float(gap.mean()),
      "middle_gap_abs_p95_mm": float(np.percentile(gap, 95)),
      "middle_over10deg_fraction": float((tilt > 10).mean()),
    }
args.output.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
