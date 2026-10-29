"""Compact commanded-versus-measured velocity plots for V033 videos."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _plot_points(
  values: np.ndarray,
  axis: int,
  limit: float,
  center: float,
  amplitude: float,
  x_values: np.ndarray,
) -> list[tuple[float, float]]:
  clipped = np.clip(values[:, axis], -limit, limit)
  y_values = center - clipped / limit * amplitude
  return list(zip(x_values.tolist(), y_values.tolist(), strict=True))


class VelocityOverlay:
  """Draw synchronized vx/vy/wz histories in the lower-right corner."""

  def __init__(
    self,
    fps: float,
    history_seconds: float = 4.0,
    limits: Sequence[float] = (0.4, 0.25, 0.5),
  ) -> None:
    self._commands: deque[np.ndarray] = deque(
      maxlen=max(2, round(fps * history_seconds))
    )
    self._actual: deque[np.ndarray] = deque(maxlen=self._commands.maxlen)
    self._limits = np.asarray(limits, dtype=np.float32)

  def draw(
    self,
    frame: np.ndarray,
    command: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
  ) -> np.ndarray:
    command_array = np.asarray(command, dtype=np.float32)
    actual_array = np.asarray(actual, dtype=np.float32)
    if command_array.shape != (3,) or actual_array.shape != (3,):
      raise ValueError("velocity overlay requires three-element vx/vy/wz vectors")
    self._commands.append(command_array.copy())
    self._actual.append(actual_array.copy())

    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = image.size
    panel_width = min(round(width * 0.30), width - 32)
    panel_height = min(round(height * 0.32), height - 32)
    x0 = width - panel_width - 16
    y0 = height - panel_height - 16
    try:
      label_font = ImageFont.truetype("DejaVuSans.ttf", 17)
      value_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
      label_font = ImageFont.load_default()
      value_font = label_font
    draw.rounded_rectangle(
      (x0, y0, width - 16, height - 16),
      radius=8,
      fill=(8, 12, 18, 178),
      outline=(255, 255, 255, 80),
      width=1,
    )

    commands = np.asarray(self._commands)
    actual_values = np.asarray(self._actual)
    labels = ("vx", "vy", "wz")
    colors = ((255, 102, 102, 255), (102, 220, 140, 255), (95, 170, 255, 255))
    left = x0 + 42
    right = width - 24
    row_height = (panel_height - 12) / 3
    for axis, (label, color) in enumerate(zip(labels, colors, strict=True)):
      top = y0 + 6 + axis * row_height
      bottom = top + row_height - 4
      center = (top + bottom) / 2
      draw.line((left, center, right, center), fill=(255, 255, 255, 60), width=1)
      draw.text((x0 + 8, top + 3), label, fill=color, font=label_font)
      draw.text(
        (x0 + 8, top + 22),
        f"{command_array[axis]:+.2f}/{actual_array[axis]:+.2f}",
        fill=(235, 235, 235, 255),
        font=value_font,
      )
      limit = max(
        float(self._limits[axis]),
        float(np.max(np.abs(commands[:, axis]))) * 1.1,
        float(np.max(np.abs(actual_values[:, axis]))) * 1.1,
      )
      count = len(actual_values)
      if count == 1:
        x_values = np.asarray([right], dtype=np.float32)
      else:
        x_values = np.linspace(left, right, count)

      command_points = _plot_points(
        commands, axis, limit, center, row_height * 0.38, x_values
      )
      for index in range(0, max(0, count - 1), 6):
        end = min(index + 3, count - 1)
        if end > index:
          draw.line(command_points[index : end + 1], fill=(*color[:3], 150), width=1)
      if count > 1:
        draw.line(
          _plot_points(actual_values, axis, limit, center, row_height * 0.38, x_values),
          fill=color,
          width=2,
        )

    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))
