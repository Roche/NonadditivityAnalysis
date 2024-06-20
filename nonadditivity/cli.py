"""Entry point for nonadditivity package."""

import click

from nonadditivity import __version__ as version
from nonadditivity.nonadditivity_workflow import run_nonadd_calculation
from nonadditivity.utils.commandline import InputOptions, add_entry_point_options
from nonadditivity.utils.log import get_logger, setup_logging

logger = get_logger()


@click.command("Nonadditivity CLI", context_settings={"show_default": True})
@click.version_option(version=version)
@add_entry_point_options
def main(input_options: InputOptions) -> None:
    """Entry point for the nonnaditivity command line tool.

    Nonadditivity Analysis based on matched MMPs

    Args:
      input_options (InputOptions): parameters for the program.
    """
    setup_logging(level=input_options.verbose, out_file=input_options.log_file)
    run_nonadd_calculation(input_options=input_options)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
