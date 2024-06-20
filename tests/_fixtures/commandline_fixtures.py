from typing import Any

import pytest

from nonadditivity.utils.commandline import InputOptions


@pytest.fixture()
def input_option_arguments() -> dict[str, Any]:
    return {
        "infile_": "tests/_test_files/test_pchembl_input.txt",
        "update_": True,
        "max_heavy_": 32,
        "no_chiral_": True,
        "delimiter_": "comma",
        "include_censored_": False,
        "classify_": False,
        "canonicalize_": False,
        "max_heavy_in_transformation_": 10,
        "series_column_": "test_series",
        "property_columns_": ("prop1", "prop2"),
        "units_": ("noconv", "M"),
        "directory_": None,
        "verbose_": 0,
        "log_file_": None,
    }


@pytest.fixture()
def input_option_keyword_arguments_nounits() -> dict[str, Any]:
    return {
        "infile_": "tests/_test_files/test_pchembl_input.txt",
        "update_": True,
        "max_heavy_": 32,
        "no_chiral_": True,
        "canonicalize_": False,
        "delimiter_": "comma",
        "include_censored_": False,
        "max_heavy_in_transformation_": 10,
        "series_column_": "test_series",
        "property_columns_": ("prop1", "prop2"),
        "classify_": False,
        "units_": (),
        "directory_": None,
        "verbose_": 2,
        "log_file_": None,
    }


@pytest.fixture()
def cli_input_arguments() -> list[str]:
    return [
        "--infile",
        "tests/_test_files/test_pchembl_input.txt",
        "--delimiter",
        "tab",
        "--series-column",
        "TEST_SERIES_ID",
        "--property-columns",
        "TEST_PCHEMBL_VALUE",
        "--max-heavy-in-transformation",
        "16",
    ]


@pytest.fixture()
def input_options_update(paths: dict[str, str]) -> InputOptions:
    return InputOptions(
        infile_=paths["test_input"],
        update_=True,
        max_heavy_=70,
        no_chiral_=False,
        delimiter_="tab",
        include_censored_=False,
        classify_=False,
        directory_=None,
        canonicalize_=True,
        max_heavy_in_transformation_=16,
        series_column_="TEST_SERIES_ID",
        property_columns_=("TEST_PCHEMBL_VALUE",),
        units_=(),
        verbose_=2,
        log_file_=None,
    )


@pytest.fixture()
def input_options(paths: dict[str, str]) -> InputOptions:
    return InputOptions(
        infile_=paths["test_input"],
        update_=False,
        max_heavy_=70,
        no_chiral_=False,
        delimiter_="tab",
        canonicalize_=True,
        classify_=False,
        include_censored_=False,
        max_heavy_in_transformation_=16,
        directory_=None,
        series_column_=None,
        property_columns_=("TEST_PCHEMBL_VALUE",),
        units_=(),
        verbose_=2,
        log_file_=paths["temp_log_file"],
    )


@pytest.fixture()
def input_options_multiprops(paths: dict[str, str]) -> InputOptions:
    return InputOptions(
        infile_=paths["test_input"],
        update_=False,
        max_heavy_=70,
        no_chiral_=False,
        delimiter_="tab",
        include_censored_=False,
        directory_=None,
        canonicalize_=True,
        classify_=False,
        max_heavy_in_transformation_=16,
        series_column_=None,
        property_columns_=(
            "TEST_PCHEMBL_VALUE",  # type: ignore
            "TEST_PCHEMBL_VALUE2",
        ),
        units_=(),
        verbose_=2,
        log_file_=paths["temp_log_file"],
    )


@pytest.fixture()
def input_options_multiprops_censored(paths: dict[str, str]) -> InputOptions:
    return InputOptions(
        infile_=paths["test_input"],
        update_=False,
        max_heavy_=70,
        no_chiral_=False,
        delimiter_="tab",
        include_censored_=True,
        canonicalize_=True,
        directory_=None,
        classify_=False,
        max_heavy_in_transformation_=16,
        series_column_=None,
        property_columns_=(
            "TEST_PCHEMBL_VALUE",  # type: ignore
            "TEST_PCHEMBL_VALUE2",
        ),
        units_=(),
        verbose_=2,
        log_file_=paths["temp_log_file"],
    )


@pytest.fixture()
def input_test_values() -> dict[str, str | int | bool]:
    return {
        "max_heavy": 32,
        "no_chiral": True,
        "delimiter": "comma",
        "include_censored": False,
        "series_column": "test_series",
    }
