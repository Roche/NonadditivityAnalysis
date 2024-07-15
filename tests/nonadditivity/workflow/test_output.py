"""Testing nonadditivity.workflow.output."""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from nonadditivity.workflow.nonadditivity_core import (
    add_mean_and_std_na_to_df,
    calculate_na_output,
)
from nonadditivity.workflow.output import (
    _write_c2c_file,
    _write_naa_file,
    _write_per_compound_file,
    _write_std_to_cmdline,
    write_output_files,
    write_smiles_id_file,
)
from tests._utils import assert_exists_and_remove, same_size


def test_write_smiles_id_file(
    smiles_dataframe: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Test nonadditivity.workflow.output:write_smiles_id_file.

    Args:
        smiles_dataframe (pd.DataFrame): smiles to write
        paths (dict[str, Path]): paths for testing.
    """
    write_smiles_id_file(
        per_compound_dataframe=smiles_dataframe,
        smifile=paths["temp_smiles"],
    )
    data = pd.read_table(paths["temp_smiles"], names=["SMILES", "Compound_ID"])
    data = data.set_index("Compound_ID", drop=False)
    assert_exists_and_remove(path=paths["temp_smiles"])


@pytest.mark.parametrize(
    "cols, cens, outpath",
    [
        (["TEST_PCHEMBL_VALUE"], False, "test_c2c"),
        (
            ["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"],
            False,
            "test_c2c_mult",
        ),
        (
            ["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"],
            True,
            "test_c2c_mult_wc",
        ),
    ],
)
def test_write_c2c_file(
    circles: list[list[str]],
    paths: dict[str, Path],
    per_cpd_dataframe: pd.DataFrame,
    cols: list[str],
    cens: bool,
    outpath: str,
) -> None:
    """Test nonadditivity.workflow.output:write_c2c_file.

    Args:
        circles (list[list[str]]): circles to write
        paths (dict[str, Path]): paths for writing
        per_cpd_dataframe (pd.DataFrame): compounds
        cols (list[str]): columns in compound dataframe
        cens (bool): whether to include censored values
        outpath (str): key for validation output.
    """
    _, _, _, c2c_df = calculate_na_output(
        circles=circles,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=cols,
        series_column=None,
        include_censored=cens,
    )
    _write_c2c_file(c2c_dataframe=c2c_df, c2c_file_path=paths["temp_c2c"])
    assert same_size(paths["temp_c2c"], paths[outpath], rel=0.05)
    assert_exists_and_remove(path=paths["temp_c2c"])


def test_write_naa_file(
    nondadditivity_dataframe: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Test nonadditivity.workflow.output:write_naa_file.

    Args:
        nondadditivity_dataframe (pd.DataFrame): nonadditivity dataframe
        paths (dict[str, Path]): test paths.
    """
    _write_naa_file(
        path=paths["temp_naa"],
        na_dataframe=nondadditivity_dataframe,
    )
    assert_exists_and_remove(path=paths["temp_naa"])


@pytest.mark.parametrize(
    "cols, cens, outpath",
    [
        (
            ["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"],
            False,
            "test_per_cpd_mult",
        ),
        (
            ["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"],
            True,
            "test_per_cpd_mult_wc",
        ),
    ],
)
def test_write_per_compound_file(
    per_cpd_dataframe: pd.DataFrame,
    paths: dict[str, Path],
    circles: list[list[str]],
    cols: list[str],
    cens: bool,
    outpath: str,
) -> None:
    """Test nonadditivity.workflow.output:write_per_compound_file.

    Args:
        per_cpd_dataframe (pd.DataFrame): compounds
        paths (dict[str, Path]): paths for writing
        circles (list[list[str]]): circles to write
        cols (list[str]): columns in compound dataframe
        cens (bool): whether to include censored values
        outpath (str): key for validation output.
    """
    _ = calculate_na_output(
        circles=circles,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=cols,
        include_censored=cens,
    )
    add_mean_and_std_na_to_df(
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=cols,
    )
    if len(cols) == 1:
        per_cpd_dataframe = per_cpd_dataframe.drop(
            columns=["TEST_PCHEMBL_VALUE2_Censors"],
        )
    _write_per_compound_file(
        path=paths["temp_per_cpd"],
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=cols,
        include_censored=cens,
    )

    assert same_size(paths["temp_per_cpd"], paths[outpath])
    assert_exists_and_remove(path=paths["temp_per_cpd"])


@pytest.mark.parametrize(
    "ser, out",
    [
        (
            None,
            [
                "\n\nEstimated Experimental Uncertainty for property "
                "'TEST_PCHEMBL_VALUE' based on 2 nonadditivity circles:\n",
            ],
        ),
        (
            "SERIES",
            [
                "\n\nEstimated Experimental Uncertainty for property "
                "'TEST_PCHEMBL_VALUE' and series '1' based on 2 nonadditivity "
                "circles:\n",
            ],
        ),
        (
            "SERIES",
            [
                (
                    "There were not sufficient nonadditivity circles found"
                    " to make a prediction on assay uncertainty for property "
                    "'TEST_PCHEMBL_VALUE'!"
                ),
            ],
        ),
    ],
)
def test_write_std_to_cmdline(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
    caplog: pytest.LogCaptureFixture,
    ser: Literal["SERIES"] | None,
    out: list[str],
) -> None:
    """Test nonadditivity.workflow.output:write_std_to_cmdline.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): circles
        caplog (pytest.LogCaptureFixture): log output of test
        ser (Literal['SERIES'] | None): series column
        out (list[str]): output expected in log.
    """
    naa_df, _, _, _ = calculate_na_output(
        circles,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=["TEST_PCHEMBL_VALUE"],
        series_column=ser,
        include_censored=False,
    )
    if "sufficient" not in out[0]:
        naa_df["Series"] = ["1" for _ in range(len(naa_df))]
    caplog.set_level(logging.INFO)
    _write_std_to_cmdline(
        property_columns=["TEST_PCHEMBL_VALUE"],
        na_dataframe=naa_df,
        series_column=ser,
    )
    for message in out:
        assert message in caplog.text


@pytest.mark.parametrize(
    ("cols, series, outpath"),
    [
        (["TEST_PCHEMBL_VALUE"], None, ""),
        (["TEST_PCHEMBL_VALUE"], "Series", "_series"),
        (["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"], None, "_mult"),
    ],
)
def test_write_ouput(
    per_cpd_dataframe: pd.DataFrame,
    paths: dict[str, Path],
    circles: list[list[str]],
    cols: list[str],
    series: str | None,
    outpath: str,
) -> None:
    """Test nonadditivity.workflow.output:write_output.

    Args:
        per_cpd_dataframe (pd.DataFrame): compound data
        paths (dict[str, Path]): test paths
        circles (list[list[str]]): circles
        cols (list[str]): columns in compound data
        series (str | None): name of series column
        outpath (str): key for path to check
    """
    nadf, _, _, c2c_dataframe = calculate_na_output(
        circles=circles,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=cols,
        series_column=series,
        include_censored=False,
    )
    add_mean_and_std_na_to_df(
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=cols,
        series_column=series,
    )

    write_output_files(
        per_compound_dataframe=per_cpd_dataframe,
        na_dataframe=nadf,
        c2c_dataframe=c2c_dataframe,
        c2c_file_path=paths["temp_c2c"],
        per_compound_path=paths["temp_per_cpd"],
        naa_file_path=paths["temp_naa"],
        property_columns=cols,
        series_column=series,
        canonicalize=True,
        include_censored=False,
    )
    for new, testfile in zip(
        [
            paths["temp_naa"],
            paths["temp_per_cpd"],
            paths["temp_c2c"],
        ],
        [
            paths[f"test_naa{outpath}"],
            paths[f"test_per_cpd{outpath}"],
            paths[f"test_c2c{outpath}"],
        ],
    ):
        assert same_size(new, testfile, rel=0.5)
        assert_exists_and_remove(new)
    assert_exists_and_remove(paths["temp_canonical_transf"])
    assert_exists_and_remove(paths["temp_canonical_naa"])
