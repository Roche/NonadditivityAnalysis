"""Handle commanline interface with click.

This file handles all the commandline input parsing for
the NonadditivityAnalysis package using click.

"""

import logging
import os
from collections.abc import Callable
from pathlib import Path

import click

VERBOSITY_LEVEL = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def add_entry_point_options(command: Callable) -> Callable:
    """Handle command line input.

    Adds all needed flags to an option
    that can then be run in the command line and parses them into an
    InputOptions file, that does the validity checks for the input.

    Args:
        command (Callable): function to add the command line options to.

    Returns:
        Callable: function with the command line options.
    """

    def set_click_attrs(dest: Callable, src: Callable) -> None:
        """Add __name__, __doc__ and __click_params__ from src to destination.

        Args:
            dest (Callable): function where src' attributes are copied to
            src (Callable): attributes from this function are copoied to dest
        """
        dest.__name__ = src.__name__
        dest.__doc__ = src.__doc__
        dest.__click_params__ = src.__click_params__  # type: ignore

    param_names = []

    def add_option(*args, **kwargs) -> None:
        """Add input click.options to function."""
        param_names.append(args[-1].lstrip("-").replace("-", "_"))
        click.option(*args, **kwargs)(command)

    add_option(
        "-i",
        "--infile",
        "infile_",
        help="Input file",
        type=click.Path(exists=True, resolve_path=True, path_type=Path),
        prompt="Path to the input .txt file",
        required=True,
    )

    add_option(
        "--update",
        "update_",
        is_flag=True,
        default=False,
        type=bool,
        help="Use fragmentation and images from previous run",
    )

    add_option(
        "-m",
        "--max-heavy",
        "max_heavy_",
        help="Maximum number of heavy atoms per ligand",
        type=int,
        default=70,
    )

    add_option(
        "--max-heavy-in-transformation",
        "max_heavy_in_transformation_",
        help=(
            "Maximum number of heavy atoms that are allowed to be "
            "exchanged in a MMP."
        ),
        type=int,
        default=10,
    )

    add_option(
        "--classify",
        "classify_",
        is_flag=True,
        default=False,
        type=bool,
        help=(
            "Set this flag if you want additional properties for your"
            " Nonadditivty cycles to be calculated."
        ),
    )

    add_option(
        "--canonicalize",
        "canonicalize_",
        is_flag=True,
        default=False,
        type=bool,
        help=(
            "Set this flag if you want additional output files where"
            " every tranformation is only occuring in one way and not."
            " not the other"
        ),
    )

    add_option(
        "--no-chiral",
        "no_chiral_",
        is_flag=True,
        default=False,
        type=bool,
        help="Skip all transformations that include chiral centers",
    )

    add_option(
        "-d",
        "--delimiter",
        "delimiter_",
        help="Specify delimiter in input file",
        type=click.Choice(["tab", "space", "comma", "semicolon"]),
        default="tab",
    )

    add_option(
        "--include-censored",
        "include_censored_",
        is_flag=True,
        default=False,
        type=bool,
        help=(
            "Include Circles where at least one out of four compounds has "
            "a censored value"
        ),
    )

    add_option(
        "-o",
        "--output-directory",
        "directory_",
        default=None,
        type=click.Path(writable=True, resolve_path=True, path_type=Path),
        help="Output directory name",
    )

    add_option(
        "--series-column",
        "series_column_",
        default=None,
        type=str,
        help="Column that identifies subseries within the dataset",
    )

    add_option(
        "-p",
        "--property-columns",
        "property_columns_",
        type=str,
        multiple=True,
        default=None,
        help="Property columns for which Nonadditivity should be calculated",
    )

    add_option(
        "-u",
        "--units",
        "units_",
        multiple=True,
        default=None,
        type=click.Choice(["M", "mM", "uM", "nM", "pM", "noconv"]),
        help=(
            "Unit of the activity given. Need to supply either"
            " None or as many as property_columns."
        ),
    )

    add_option(
        "-v",
        "--verbose",
        "verbose_",
        count=True,
        default=0,
        help=(
            "use -v to set logging level to info, use -vv to set logging level"
            " to debug. without -v logging level is set to warning."
        ),
    )

    add_option(
        "--log",
        "log_file_",
        default=None,
        type=str,
        help=(
            "File to write all logging info to. "
            "If none is provided, logging is displayed in commandline. "
            "It will be placed in the output directory."
        ),
    )

    def make_input_options_wrapper(**kwargs) -> Callable:
        """Create input options instance.

        creates an input options object from the flags provided in the command
        line and parses them so the original command is only taking the
        inputoptions object but not every single command line flag.

        Returns:
            Callable: command with input options as argument.
        """
        in_kwargs = {}
        for name in param_names:
            value = kwargs.pop(name)
            in_kwargs[name] = value

        kwargs["input_options"] = InputOptions(**in_kwargs)
        return command(**kwargs)

    set_click_attrs(make_input_options_wrapper, command)
    return make_input_options_wrapper


class InputOptions:
    """Handles the type checking for the."""

    def __init__(
        self,
        infile_: Path,
        update_: bool,
        max_heavy_: int,
        max_heavy_in_transformation_: int,
        classify_: bool,
        canonicalize_: bool,
        no_chiral_: bool,
        delimiter_: str,
        include_censored_: bool,
        directory_: Path | None,
        series_column_: str | None,
        property_columns_: tuple[str, ...],
        units_: tuple[str | None, ...],
        verbose_: int,
        log_file_: str | None,
    ) -> None:
        """Create input Option instance with cli arguments.

        Args:
            infile_ (Path): Path to input file
            update_ (bool): whether to use old fragmentation
            max_heavy_ (int): max heavy atoms per ligand
            max_heavy_in_transformation_ (int): max heavy input in transformation
            classify_ (bool): whether to run classification
            canonicalize_ (bool): wheter to provide canonicalized output
            no_chiral_ (bool): whether to skip chiral transformation
            delimiter_ (str): delimiter used in input file
            include_censored_ (bool): whether to include censored values
            directory_ (Path | None): where to store intermediate and output files
            series_column_ (str | None): column containing info about subseries
            property_columns_ (tuple[str, ...]): name of the columns with properties
            to analyze
            units_ (tuple[str  |  None, ...]): units of provided proeprty columns
            verbose_ (int): whether to give verbose output
            log_file_ (str | None): file to log log messages to

        Raises:
            ValueError: Raised if any input argument is not valid.
        """
        self.infile = infile_
        self.update = update_
        self.max_heavy = max_heavy_
        self.classify = classify_
        self.canonicalize = canonicalize_
        self.max_heavy_in_transformation = max_heavy_in_transformation_
        self.no_chiral = no_chiral_
        self.delimiter = delimiter_
        self.include_censored = include_censored_
        self.directory = directory_
        self.series_column = series_column_
        self.property_columns = property_columns_
        self.units = units_
        self.verbose = verbose_
        self.log_file = log_file_
        if not self._property_columns_and_units_correct():
            raise ValueError(
                (
                    "If Units is provided, the number of units has to be the "
                    "same as the number of property_columns provided."
                ),
            )

    def _property_columns_and_units_correct(self) -> bool:
        return len(self.property_columns) == len(self.units)

    @property
    def infile(self) -> Path:
        """Path to input file for analysis."""
        return self._infile

    @infile.setter
    def infile(self, value: Path) -> None:
        self._infile = value

    @property
    def update(self) -> bool:
        """Whether to use previous fragmentation."""
        return self._update

    @update.setter
    def update(self, value: bool) -> None:
        self._update = value

    @property
    def max_heavy(self) -> int:
        """Max num heavy atoms in ligands."""
        return self._max_heavy

    @max_heavy.setter
    def max_heavy(self, value: int) -> None:
        self._max_heavy = value

    @property
    def classify(self) -> bool:
        """Whether to classify output."""
        return self._classify

    @classify.setter
    def classify(self, value: bool) -> None:
        self._classify = value

    @property
    def canonicalize(self) -> bool:
        """Whether to canonicalize output."""
        return self._canonicalize

    @canonicalize.setter
    def canonicalize(self, value: bool) -> None:
        self._canonicalize = value

    @property
    def max_heavy_in_transformation(self) -> int:
        """Max num heavy atoms in transformations."""
        return self._max_heavy_in_transformation

    @max_heavy_in_transformation.setter
    def max_heavy_in_transformation(self, value: int) -> None:
        self._max_heavy_in_transformation = value

    @property
    def no_chiral(self) -> bool:
        """Whether to use chiral transformations in analysis."""
        return self._no_chiral

    @no_chiral.setter
    def no_chiral(self, value: bool) -> None:
        self._no_chiral = value

    @property
    def delimiter(self) -> str:
        """Delimiter used in the input file."""
        return self._delimiter

    @delimiter.setter
    def delimiter(self, value: str) -> None:
        self._delimiter = value

    @property
    def include_censored(self) -> bool:
        """Whether to include censored values."""
        return self._include_censored

    @include_censored.setter
    def include_censored(self, value: bool) -> None:
        self._include_censored = value

    @property
    def directory(self) -> Path:
        """Directory for where to store output."""
        return self._directory

    @directory.setter
    def directory(self, value: Path | None) -> None:
        if value is None:
            value = Path(os.path.dirname(self.infile)).resolve()
        if not value.exists():
            os.makedirs(value)
        self._directory = value

    @property
    def series_column(self) -> str | None:
        """Name of series column if any."""
        return self._series_column

    @series_column.setter
    def series_column(self, value: str | None) -> None:
        self._series_column = value

    @property
    def property_columns(self) -> list[str]:
        """List of columns containig to analyse properties."""
        return self._property_columns

    @property_columns.setter
    def property_columns(self, value: tuple[str, ...]) -> None:
        self._property_columns = list(value)

    @property
    def units(self) -> list[str | None]:
        """List of units."""
        return self._units  # type: ignore

    @units.setter
    def units(self, value: tuple[str | None, ...]) -> None:
        if len(value) == 0:
            self._units = [None for _ in self.property_columns]
            return
        self._units = list(value)

    @property
    def verbose(self) -> int:
        """Log level."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: int) -> None:
        try:
            value = int(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                (
                    "Verbosity flag is used wrong, please use 'nonadditivity"
                    " --help' to get more info"
                ),
            ) from exc
        value = min(value, 2)
        value = max(value, 0)
        self._verbose = VERBOSITY_LEVEL[value]

    @property
    def log_file(self) -> Path | None:
        """Log file path for nonadditivity analysis."""
        return self._log_file

    @log_file.setter
    def log_file(self, value: str | None) -> None:
        if value is None:
            self._log_file = None
            return
        self._log_file = self.directory / value
