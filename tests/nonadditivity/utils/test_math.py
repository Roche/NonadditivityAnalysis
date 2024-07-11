"""Test nonadditivity.utils.math."""

import numpy as np
import pandas as pd
import pytest

from nonadditivity.utils.math import (
    _calculate_mean_and_std_per_compound,
    _calculate_nonadditivity,
    _calculate_theo_quantiles,
    is_number,
    mad_std,
    sn_medmed_std,
)


@pytest.fixture()
def int_array() -> list[int]:
    """Numbers to test functionality with.

    Returns:
        list[int]: [1,2,3,4,5]
    """
    return list(range(1, 6))


@pytest.mark.parametrize(
    "value, sol",
    [
        (2, True),
        (0, True),
        (-2, True),
        (2.321, True),
        ("3.4", True),
        ("A", False),
        (None, False),
    ],
)
def test_is_number(value: float | str | None, sol: bool) -> None:
    """Test nonadditivity.utils.math:is_numnber.

    Args:
        value (float | str | None): value to test
        sol (bool): expected solution
    """
    assert is_number(value=value) == sol


def test_mad_std(int_array: list[int]) -> None:
    """Test nonadditivity.utils.math:mad_std.

    Args:
        int_array (list[int]): input values
    """
    assert mad_std(values=int_array) == 1.4826
    assert mad_std(values=np.array(int_array)) == 1.4826  # type: ignore

    with pytest.raises(ValueError):
        mad_std(values=["A", 3])  # type: ignore
    with pytest.raises(ValueError):
        mad_std(values=None)  # type: ignore


def test_sn_medmed_std(int_array: list[int]) -> None:
    """Test nonadditivity.utils.math:sn_medmed_std.

    Args:
        int_array (list[int]): input values
    """
    assert sn_medmed_std(values=int_array) == 1.7889000000000002
    assert sn_medmed_std(values=2) == 0  # type: ignore
    with pytest.raises(ValueError):
        sn_medmed_std(values=["A", 3])  # type: ignore
    with pytest.raises(ValueError):
        sn_medmed_std(values="A")  # type: ignore
    with pytest.raises(ValueError):
        sn_medmed_std(values=None)  # type: ignore


def test_calculate_nonadditivity(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.utils.math:_calculate_nonadditivity.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circles
    """
    assert (
        _calculate_nonadditivity(
            values=[
                per_cpd_dataframe.loc[index, "TEST_PCHEMBL_VALUE"]
                for index in circles[0]
            ],
        )
        == 2.5
    )


def test_calculate_theo_quantile(int_array: list[int]) -> None:
    """Test nonadditivity.utils.math:_calculate_theo_quantiles.

    Args:
        int_array (list[int]): input values
    """
    assert np.allclose(
        _calculate_theo_quantiles(  # type: ignore
            series_columns=None,
            series_ids=None,  # type: ignore
            nonadditivities=int_array,  # type: ignore
        ),
        [
            -1.1289975352961017,
            -0.4856527065756922,
            0.0,
            0.48565270657569204,
            1.1289975352961017,
        ],
        rtol=0.01,
    )


def test_calculate_theo_quantile_series(int_array: list[int]) -> None:
    """Test nonadditivity.utils.math:_calculate_theo_quantiles.

    Args:
        int_array (list[int]): input values
    """
    int_array += [3]
    assert np.allclose(
        _calculate_theo_quantiles(
            series_columns="SERIES",
            series_ids=["a", "b", "a", "b", "b", "c"],  # type: ignore
            nonadditivities=int_array,  # type: ignore
        ),
        [
            -0.54495214,
            -0.8193286198336103,
            0.54495214,
            0.0,
            0.8193286198336103,
            0,
        ],
        rtol=0.01,
    )


def test_na_per_compound(nondadditivity_dataframe: pd.DataFrame) -> None:
    """Test nonadditivity.utils.math:_calculate_mean_and_std_per_compound.

    Args:
        nondadditivity_dataframe (pd.DataFrame): na data
    """
    assert np.array_equal(
        _calculate_mean_and_std_per_compound(  # type: ignore
            per_compound_dataframe=nondadditivity_dataframe,
            property_column="TEST_PROPERTY",
        ),
        [[2.0, 1.4142135623730951, 5]],  # type: ignore
    )
    emptydf = pd.DataFrame()
    emptydf["EMPTY_Nonadditivities"] = [[]]
    assert np.array_equal(
        _calculate_mean_and_std_per_compound(
            per_compound_dataframe=emptydf,
            property_column="EMPTY",
        ),
        [[None, None, 0]],  # type: ignore
    )


def test_na_per_compound_series(
    nondadditivity_dataframe: pd.DataFrame,
) -> None:
    """Test nonadditivity.utils.math:_calculate_mean_and_std_per_compound.

    Args:
        nondadditivity_dataframe (pd.DataFrame): na data
    """
    assert np.array_equal(
        _calculate_mean_and_std_per_compound(  # type: ignore
            per_compound_dataframe=nondadditivity_dataframe,
            property_column="TEST_PROPERTY_SERIES",
            series_column="TEST_SERIES",
        ),
        [[[2.0, 1.632993161855452, 3]], [[2.0, 1.0, 2]]],  # type: ignore
    )
