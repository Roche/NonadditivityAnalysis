"""Test nonadditivity.utils.commandline."""
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nonadditivity.utils.commandline import InputOptions, add_entry_point_options


@pytest.fixture()
def decorated_function() -> Callable[..., Any]:
    """Mimik command line entry point."""

    def function(input_options: InputOptions) -> InputOptions:
        return input_options

    return add_entry_point_options(function)


def test_click(
    decorated_function: Callable[..., Any],
    input_option_arguments: dict[str, Any],
    input_option_keyword_arguments_nounits: dict[str, Any],
) -> None:
    """Test click framework.

    Args:
        decorated_function (Callable[..., Any]): function mimiking entry point
        input_option_arguments (dict[str, Any]): input options
        input_option_keyword_arguments_nounits (dict[str, Any]): more input
        options
    """
    input_arguments = decorated_function(**input_option_arguments)
    assert input_arguments.infile.resolve().as_posix() == os.path.abspath(
        input_option_arguments["infile_"],
    )
    assert input_arguments.directory.resolve().as_posix() == os.path.abspath(
        os.path.dirname(input_option_arguments["infile_"]),
    )
    assert input_arguments.update
    assert input_arguments.verbose == logging.WARNING

    input_arguments = decorated_function(**input_option_keyword_arguments_nounits)
    assert np.array_equal(input_arguments.units, [None, None])  # type: ignore

    assert input_arguments.verbose == logging.DEBUG


def test_input_options_init(
    input_option_arguments: dict[str, Any],
    input_test_values: dict[str, str | int | bool],
    input_option_keyword_arguments_nounits: dict[str, Any],
) -> None:
    """Test init input options.

    Args:
        input_option_arguments (dict[str, Any]): input options
        input_test_values (dict[str, str  |  int  |  bool]): input options
        input_option_keyword_arguments_nounits (dict[str, Any]): input options.
    """
    io = InputOptions(**input_option_keyword_arguments_nounits)
    for key, value in input_test_values.items():
        assert getattr(io, key) == value
    io = InputOptions(**input_option_arguments)
    for key, value in input_test_values.items():
        assert getattr(io, key) == value

    with pytest.raises(ValueError):
        InputOptions(
            infile_=Path("test.file"),
            update_=False,
            max_heavy_=70,
            max_heavy_in_transformation_=10,
            no_chiral_=True,
            canonicalize_=False,
            delimiter_="tab",
            include_censored_=False,
            directory_=Path("tests/_directory"),
            series_column_="test_series",
            classify_=False,
            property_columns_=("test_propery",),
            units_=("M", None),  # type: ignore
            verbose_=0,
            log_file_=None,
        )
    with pytest.raises(ValueError):
        InputOptions(
            infile_=Path("test.file"),
            update_=False,
            max_heavy_=70,
            no_chiral_=True,
            max_heavy_in_transformation_=10,
            delimiter_="tab",
            classify_=False,
            canonicalize_=False,
            include_censored_=False,
            directory_=Path("tests/_directory"),
            series_column_="test_series",
            property_columns_=("test_propery", "ueppa"),  # type: ignore
            units_=("M",),
            verbose_=0,
            log_file_=None,
        )

    with pytest.raises(ValueError):
        InputOptions(
            infile_=Path("test.file"),
            update_=False,
            max_heavy_=70,
            no_chiral_=True,
            classify_=False,
            max_heavy_in_transformation_=10,
            delimiter_="tab",
            canonicalize_=False,
            include_censored_=False,
            directory_=Path("tests/_directory"),
            series_column_="test_series",
            property_columns_=("test_propery", "ueppa"),  # type: ignore
            units_=("M",),
            verbose_="ASDF",  # type: ignore
            log_file_=None,
        )
    os.rmdir(os.path.abspath(__file__ + "/../../../_directory"))
