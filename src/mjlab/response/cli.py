"""CLI entry point for suspended actuator response tests."""

import sys

import tyro

from mjlab.response.config import ResponseConfig
from mjlab.response.runner import run_response
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks


def main() -> None:
  """Parse the response task CLI and execute the selected protocol."""
  maybe_print_top_level_help("response")

  import mjlab.tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    ResponseConfig,
    args=remaining_args,
    default=ResponseConfig(),
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_response(task_id, cfg)


if __name__ == "__main__":
  main()
