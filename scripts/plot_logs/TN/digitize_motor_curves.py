"""Digitize the motor curves embedded in the two datasheet screenshots.

The screenshots use one torque axis and six independently scaled y-axes.  This
script identifies the coloured curve pixels, converts their pixel coordinates
with manually calibrated tick positions, writes CSV files, and redraws the
results with Matplotlib.

The result is an approximation of the source data.  Its precision is limited
by line thickness, anti-aliasing, and curves that obscure one another.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AxisCalibration:
  """Map image pixels to values using labelled tick positions."""

  pixels: tuple[float, ...]
  values: tuple[float, ...]

  def convert(self, pixels: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(self.pixels, self.values, 1)
    return slope * pixels + intercept


@dataclass(frozen=True)
class CurveSpec:
  """Colour and axis information for one curve."""

  column: str
  label: str
  color: tuple[int, int, int]
  axis: AxisCalibration
  y_range: tuple[int, int]
  fallback: str | None = None
  constant_value: float | None = None
  max_slope: float | None = None


@dataclass(frozen=True)
class ImageSpec:
  """All digitization settings for one screenshot."""

  filename: str
  x_range: tuple[int, int]
  plot_y_range: tuple[int, int]
  torque_axis: AxisCalibration
  curves: tuple[CurveSpec, ...]


CAL_3507 = {
  "vbus_v": AxisCalibration(
    (295, 392, 489, 585, 682),
    (25.0, 24.5, 24.0, 23.5, 23.0),
  ),
  "ibus_a": AxisCalibration(
    (261, 341, 420, 500, 580, 660, 740),
    (6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0),
  ),
  "speed_rpm": AxisCalibration(
    (257, 323, 389, 455, 522, 588, 654, 720),
    (150.0, 147.5, 145.0, 142.5, 140.0, 137.5, 135.0, 132.5),
  ),
  "pin_w": AxisCalibration(
    (274, 341, 407, 474, 540, 607, 673, 740),
    (140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0, 0.0),
  ),
  "pout_w": AxisCalibration(
    (266, 328, 390, 453, 516, 578, 641, 703),
    (40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0),
  ),
  "efficiency_pct": AxisCalibration(
    (279, 347, 415, 483, 551, 619, 687),
    (60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0),
  ),
}

CAL_4310 = {
  "vbus_v": AxisCalibration(
    (264, 376, 488, 600, 712, 823),
    (25.0, 20.0, 15.0, 10.0, 5.0, 0.0),
  ),
  "ibus_a": AxisCalibration(
    (349, 460, 571, 683, 794),
    (5.0, 4.0, 3.0, 2.0, 1.0),
  ),
  "speed_rpm": AxisCalibration(
    (335, 456, 579, 702, 823),
    (200.0, 150.0, 100.0, 50.0, 0.0),
  ),
  "pin_w": AxisCalibration(
    (341, 436, 529, 624, 718, 812),
    (120.0, 100.0, 80.0, 60.0, 40.0, 20.0),
  ),
  "pout_w": AxisCalibration(
    (282, 363, 445, 525, 607, 688, 770),
    (70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0),
  ),
  "efficiency_pct": AxisCalibration(
    (346, 463, 579, 695, 811),
    (60.0, 50.0, 40.0, 30.0, 20.0),
  ),
}

IMAGE_SPECS = (
  ImageSpec(
    filename="DM-J3507-2EC-T-N.png",
    x_range=(251, 854),
    plot_y_range=(232, 744),
    torque_axis=AxisCalibration(
      (226, 342, 457, 572, 688, 803),
      (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
    ),
    curves=(
      CurveSpec(
        "vbus_v",
        "Bus voltage",
        (33, 121, 29),
        CAL_3507["vbus_v"],
        (480, 496),
        constant_value=24.0,
      ),
      # The cyan line is almost completely hidden by the purple curve.
      CurveSpec(
        "ibus_a",
        "Bus current",
        (50, 190, 180),
        CAL_3507["ibus_a"],
        (245, 735),
        "pout_w",
      ),
      CurveSpec(
        "speed_rpm",
        "Speed",
        (10, 5, 205),
        CAL_3507["speed_rpm"],
        (245, 730),
        max_slope=20.0,
      ),
      CurveSpec(
        "pin_w",
        "Input power",
        (174, 189, 45),
        CAL_3507["pin_w"],
        (245, 735),
      ),
      CurveSpec(
        "pout_w",
        "Output power",
        (160, 20, 160),
        CAL_3507["pout_w"],
        (245, 735),
      ),
      CurveSpec(
        "efficiency_pct",
        "Efficiency",
        (200, 15, 30),
        CAL_3507["efficiency_pct"],
        (245, 735),
      ),
    ),
  ),
  ImageSpec(
    filename="DM-J4310-2EC-T-N.png",
    x_range=(498, 1197),
    plot_y_range=(259, 852),
    torque_axis=AxisCalibration(
      (479, 626, 772, 917, 1064, 1209),
      (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    ),
    curves=(
      CurveSpec(
        "vbus_v",
        "Bus voltage",
        (76, 160, 74),
        CAL_4310["vbus_v"],
        (279, 294),
        constant_value=24.0,
      ),
      CurveSpec(
        "ibus_a",
        "Bus current",
        (73, 191, 179),
        CAL_4310["ibus_a"],
        (275, 835),
        "pin_w",
      ),
      CurveSpec(
        "speed_rpm",
        "Speed",
        (49, 79, 169),
        CAL_4310["speed_rpm"],
        (275, 835),
        max_slope=5.0,
      ),
      CurveSpec(
        "pin_w",
        "Input power",
        (145, 48, 156),
        CAL_4310["pin_w"],
        (275, 835),
      ),
      CurveSpec(
        "pout_w",
        "Output power",
        (169, 181, 38),
        CAL_4310["pout_w"],
        (275, 835),
      ),
      CurveSpec(
        "efficiency_pct",
        "Efficiency",
        (237, 34, 40),
        CAL_4310["efficiency_pct"],
        (275, 835),
      ),
    ),
  ),
)

CSV_COLUMNS = (
  "torque_nm",
  "vbus_v",
  "ibus_a",
  "speed_rpm",
  "pin_w",
  "pout_w",
  "efficiency_pct",
)


def extract_pixel_centers(
  image: np.ndarray,
  spec: ImageSpec,
  curve: CurveSpec,
  threshold: float = 55.0,
) -> np.ndarray:
  """Return the subpixel y centre of a coloured curve in every x column."""
  x_min, x_max = spec.x_range
  y_min, y_max = curve.y_range
  crop = image[y_min : y_max + 1, x_min : x_max + 1].astype(float)
  distance = np.linalg.norm(crop - np.asarray(curve.color), axis=2)
  # Keep only pixels close to the best colour match in each column.  Merely
  # thresholding every match lets blended pixels at curve intersections pull
  # the centre towards another curve and creates artificial spikes.
  best_distance = distance.min(axis=0)
  best_rows = distance.argmin(axis=0)
  row_indices = np.arange(distance.shape[0])[:, None]
  same_component = np.abs(row_indices - best_rows[None, :]) <= 4
  matched = distance <= np.minimum(best_distance + 18.0, threshold)[None, :]
  matched &= same_component
  matched[:, best_distance > threshold] = False
  weights = 1.0 / (1.0 + distance)
  weights *= matched

  local_rows = np.arange(y_min, y_max + 1, dtype=float)[:, None]
  totals = weights.sum(axis=0)
  centers = np.full(x_max - x_min + 1, np.nan)
  visible = totals > 0.0
  centers[visible] = (weights[:, visible] * local_rows).sum(axis=0)
  centers[visible] /= totals[visible]
  return centers


def reject_path_jumps(values: np.ndarray, max_slope: float | None) -> np.ndarray:
  """Reject colour matches that jump implausibly far from the tracked path."""
  if max_slope is None:
    return values
  result = values.copy()
  known = np.flatnonzero(~np.isnan(result))
  if known.size == 0:
    return result
  previous = known[0]
  for index in known[1:]:
    gap = index - previous
    allowed_jump = 8.0 + max_slope * gap
    if abs(result[index] - result[previous]) > allowed_jump:
      result[index] = np.nan
    else:
      previous = index
  return result


def fill_missing(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
  """Fill obscured pixels by fallback geometry, then linear interpolation."""
  result = values.copy()
  if fallback is not None:
    use_fallback = np.isnan(result) & ~np.isnan(fallback)
    result[use_fallback] = fallback[use_fallback]

  known = np.flatnonzero(~np.isnan(result))
  if known.size < 2:
    raise ValueError("Too few curve pixels were detected")
  missing = np.flatnonzero(np.isnan(result))
  result[missing] = np.interp(missing, known, result[known])
  return result


def digitize(image_path: Path, spec: ImageSpec) -> dict[str, np.ndarray]:
  """Extract and calibrate all curves from one image."""
  image = np.asarray(Image.open(image_path).convert("RGB"))
  expected_height, expected_width = image.shape[:2]
  if expected_height <= spec.plot_y_range[1] or expected_width <= spec.x_range[1]:
    raise ValueError(f"Unexpected image size for {image_path}: {image.shape[:2]}")

  raw_pixels = {
    curve.column: extract_pixel_centers(image, spec, curve) for curve in spec.curves
  }
  x_pixels = np.arange(spec.x_range[0], spec.x_range[1] + 1, dtype=float)
  result = {"torque_nm": spec.torque_axis.convert(x_pixels)}

  # Resolve non-fallback curves first so a hidden curve can reuse their shape.
  ordered = sorted(spec.curves, key=lambda curve: curve.fallback is not None)
  filled_pixels: dict[str, np.ndarray] = {}
  for curve in ordered:
    fallback = filled_pixels.get(curve.fallback) if curve.fallback else None
    raw_curve = reject_path_jumps(raw_pixels[curve.column], curve.max_slope)
    if curve.constant_value is not None:
      visible = raw_curve[~np.isnan(raw_curve)]
      if visible.size == 0:
        raise ValueError(f"No pixels detected for {curve.column}")
      centers = np.full_like(raw_curve, np.median(visible))
    else:
      centers = fill_missing(raw_curve, fallback)
    filled_pixels[curve.column] = centers
    if curve.constant_value is None:
      result[curve.column] = curve.axis.convert(centers)
    else:
      result[curve.column] = np.full_like(centers, curve.constant_value)
  return result


def write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
  """Write aligned digitized samples to a CSV file."""
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(CSV_COLUMNS)
    rows = zip(*(data[column] for column in CSV_COLUMNS), strict=True)
    writer.writerows(tuple(f"{value:.6f}" for value in row) for row in rows)


def redraw(path: Path, title: str, data: dict[str, np.ndarray]) -> None:
  """Create a clear four-panel Matplotlib rendering of digitized data."""
  torque = data["torque_nm"]
  colors = {
    "vbus_v": "#21791d",
    "ibus_a": "#32bdb1",
    "speed_rpm": "#203eb5",
    "pin_w": "#92309c",
    "pout_w": "#a9b526",
    "efficiency_pct": "#ed2228",
  }
  fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
  voltage_axis = axes[0, 0]
  current_axis = voltage_axis.twinx()
  voltage_axis.plot(torque, data["vbus_v"], color=colors["vbus_v"], lw=2)
  current_axis.plot(torque, data["ibus_a"], color=colors["ibus_a"], lw=2)
  voltage_axis.set_ylabel("Bus voltage (V)", color=colors["vbus_v"])
  current_axis.set_ylabel("Bus current (A)", color=colors["ibus_a"])
  voltage_axis.set_title("Electrical")

  axes[0, 1].plot(
    torque,
    data["speed_rpm"],
    color=colors["speed_rpm"],
    lw=2,
  )
  axes[0, 1].set_ylabel("Speed (rpm)")
  axes[0, 1].set_title("Speed")

  axes[1, 0].plot(
    torque,
    data["pin_w"],
    color=colors["pin_w"],
    label="Input power",
    lw=2,
  )
  axes[1, 0].plot(
    torque,
    data["pout_w"],
    color=colors["pout_w"],
    label="Output power",
    lw=2,
  )
  axes[1, 0].set_ylabel("Power (W)")
  axes[1, 0].set_title("Power")
  axes[1, 0].legend()

  axes[1, 1].plot(
    torque,
    data["efficiency_pct"],
    color=colors["efficiency_pct"],
    lw=2,
  )
  axes[1, 1].set_ylabel("Efficiency (%)")
  axes[1, 1].set_title("Efficiency")

  for axis in axes.flat:
    axis.set_xlabel("Torque (N m)")
    axis.grid(alpha=0.3)
  fig.suptitle(f"Digitized motor curves: {title}", fontsize=15)
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=Path(__file__).resolve().parent,
    help="Directory containing the source PNG files",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).resolve().parent,
    help="Directory for digitized CSV and redrawn PNG files",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  for spec in IMAGE_SPECS:
    source = args.input_dir / spec.filename
    stem = source.stem
    data = digitize(source, spec)
    csv_path = args.output_dir / f"{stem}_digitized.csv"
    plot_path = args.output_dir / f"{stem}_redrawn.png"
    write_csv(csv_path, data)
    redraw(plot_path, stem, data)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
  main()
