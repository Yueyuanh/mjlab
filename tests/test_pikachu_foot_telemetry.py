from __future__ import annotations

from mjlab.tasks.pikachu_foot_telemetry import (
  ANKLE_BODY_NAMES,
  _ordered_ankle_body_ids,
)
from mjlab.utils.lab_api.string import resolve_matching_names


class _Entity:
  body_names = ("right_ankle_link", "base_link", "left_ankle_link")

  def find_bodies(self, names, preserve_order=False):
    return resolve_matching_names(names, self.body_names, preserve_order)


def test_taxel_ankle_body_ids_follow_left_right_order() -> None:
  entity = _Entity()

  body_ids = _ordered_ankle_body_ids(entity)  # type: ignore[arg-type]

  assert body_ids == [2, 0]
  assert tuple(entity.body_names[index] for index in body_ids) == ANKLE_BODY_NAMES