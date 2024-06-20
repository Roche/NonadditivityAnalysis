"""Test nonadditivity.workflows.nonadditivity_core."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nonadditivity.utils.math import _calculate_nonadditivity
from nonadditivity.workflow.input_parsing import _read_in_data
from nonadditivity.workflow.mmpdb_helper import read_raw_mmps
from nonadditivity.workflow.nonadditivity_core import (
    _create_circle_ids,
    _find_circles,
    _get_faulty_entries,
    _get_tranformation_pairs,
    _is_same_series,
    _randomize_circles,
    _update_na_per_compound,
    add_mean_and_std_na_to_df,
    add_neighbor_dictionary_to_df,
    calculate_na_output,
    get_circles,
    run_nonadditivity_core,
    top_distance_changes,
)


@pytest.fixture()
def naa_cols() -> list[str]:
    """Columns of the nonadditivity dataframe."""
    return [
        "Compound1",
        "Compound2",
        "Compound3",
        "Compound4",
        "SMILES1",
        "SMILES2",
        "SMILES3",
        "SMILES4",
        "Prop_Cpd1",
        "Prop_Cpd2",
        "Prop_Cpd3",
        "Prop_Cpd4",
        "Transformation1",
        "Transformation2",
        "Property",
        "Series",
        "Nonadditivity",
        "Theo_Quantile",
        "Circle_ID",
    ]


@pytest.fixture()
def means() -> list[list[float]]:
    """Expected mean values."""
    return [
        [
            np.mean([0.8, -2.5]),
            2.5,
            2.5,
            np.mean([-0.8, -2.5]),
            -0.8,
            0.8,
            np.nan,
            np.nan,
        ],
        [
            np.mean([-0.8, -2.5]),
            2.5,
            2.5,
            np.mean([0.8, -2.5]),
            0.8,
            -0.8,
            np.nan,
            np.nan,
        ],
    ]


@pytest.fixture()
def stds() -> list[list[float]]:
    """Expected standard deviations."""
    return [
        [
            np.std([0.8, -2.5]),
            0.0,
            0.0,
            np.std([-0.8, -2.5]),
            0.0,
            0.0,
            np.nan,
            np.nan,
        ],
        [
            np.std([-0.8, -2.5]),
            0.0,
            0.0,
            np.std([0.8, -2.5]),
            0.0,
            0.0,
            np.nan,
            np.nan,
        ],
        [
            np.std([0.8, 2.5]),
            0.0,
            0.0,
            np.std([-0.8, 2.5]),
            0.0,
            0.0,
            np.nan,
            np.nan,
        ],
        [
            np.std([-0.8, 2.5]),
            0.0,
            0.0,
            np.std([-0.8, -2.5]),
            0.0,
            0.0,
            np.nan,
            np.nan,
        ],
    ]


@pytest.mark.parametrize(
    "sm, val",
    [
        (("[*:1]Cc1ccccc1CC[*:2]", "[*:1]Cc1ccncc1CC[*:2]"), False),
        (("[*:1]Cc1ccccc1CC[*:2]", "[*:1]COCC[*:2]"), True),
        (
            ("[*:1]Cc1ccccc1CC([*:2])([*:3])", "[*:1]Cc1ccncc1CC([*:2])([*:3])"),
            False,
        ),
        (("[*:1]Cc1ccccc1CC([*:2])([*:3])", "[*:1]CCOC([*:2])([*:3])"), True),
    ],
)
def test_top_distance_changes(
    sm: tuple[str, str],
    val: bool,
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:top_distance_changes.

    Args:
        sm (tuple[str,str]): lhs and rhs smiles
        val (bool): solution
    """
    assert (
        top_distance_changes(
            var_lhs=sm[0],
            var_rhs=sm[1],
        )
        == val
    )


def test_add_neighbor_dictionary_to_df(
    paths: dict[str, Path],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:add_neighbor_dict_to_df.

    Args:
        paths (dict[str, Path]): paths to solution dataframes.
    """
    mmp_dataframe = read_raw_mmps(paths["test_mmp"])
    data = _read_in_data(
        paths["test_input"],
        series_column="TEST_SERIES_ID",
        property_columns=["TEST_PCHEMBL_VALUE"],
    )
    chiral_df = pd.DataFrame()
    for value, key in zip(
        ["CCC", "CCH", "C1", "C2", "CC>>C[@@H]", "CF"],
        [
            "SMILES_LHS",
            "SMILES_RHS",
            "ID_LHS",
            "ID_RHS",
            "TRANSFORMATION",
            "CONSTANT",
        ],
    ):
        chiral_df[key] = [value]
    mmp_dataframe = pd.concat([mmp_dataframe, chiral_df])
    add_neighbor_dictionary_to_df(
        mmp_dataframe=mmp_dataframe,
        per_compound_dataframe=data,
        no_chiral=True,
    )
    assert ("ID2" and "ID3" and "ID4" and "ID5" and "ID7") in data.loc[
        "ID1",
        "Neighbor_dict",
    ]
    assert data.loc["ID1", "Neighbor_dict"]["ID2"][0] == "[*:1][H]>>[*:1]F"
    assert data.loc["ID1", "Neighbor_dict"]["ID2"][1] == "[*:1]C(C)=Cc1ccc(C)nc1"


def test_get_transformation_pairs(per_cpd_dataframe: pd.DataFrame) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:get_transformation_pairs.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
    """
    neighbordict = dict(
        zip(
            per_cpd_dataframe.index.to_numpy(),
            per_cpd_dataframe.Neighbor_dict.to_numpy(),
        ),
    )

    transf_pairs = _get_tranformation_pairs(neighbordict=neighbordict)
    for key, n_dict in zip(
        per_cpd_dataframe.index.to_numpy(),
        per_cpd_dataframe.Neighbor_dict.to_numpy(),
    ):
        for neighbor, transformation in n_dict.items():
            assert (key, neighbor) in transf_pairs[transformation]


def test_find_circles(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:find_circles.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): solution
    """
    neighbordict = {
        key: {k: v[0] for k, v in value.items()}
        for key, value in zip(
            per_cpd_dataframe.index.to_numpy(),
            per_cpd_dataframe.Neighbor_dict.to_numpy(),
        )
    }
    tp = _get_tranformation_pairs(neighbordict=neighbordict)
    circle = _find_circles(neighbordict=neighbordict, transformation_pairs=tp)
    assert np.array_equal(circle, circles)  # type: ignore


def test_get_circles(per_cpd_dataframe: pd.DataFrame, circles: list[list[str]]) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:get_circles.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): solution
    """
    neighbordict = {
        key: {k: v[0] for k, v in value.items()}
        for key, value in zip(
            per_cpd_dataframe.index.to_numpy(),
            per_cpd_dataframe.Neighbor_dict.to_numpy(),
        )
    }
    tp = _get_tranformation_pairs(neighbordict=neighbordict)
    circle, tp2 = get_circles(per_compound_dataframe=per_cpd_dataframe)
    assert np.array_equal(circle, circles)  # type: ignore
    assert tp == tp2


def test_randomize_circles(circles: list[list[str]]) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:randomize_circles.

    Args:
        circles (list[list[str]]): list of circles
    """
    randomized_circles, min_ind = _randomize_circles(circles=circles)
    for index, randomized_circle in enumerate(randomized_circles):
        assert (
            randomized_circle
            == (circles[index] + circles[index])[min_ind[index] : min_ind[index] + 4]
        )


def test_create_circle_ids(circles: list[list[str]]) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:create_circle_ids.

    Args:
        circles (list[list[str]]): list of circles.
    """
    for circle in circles:
        assert (
            _create_circle_ids(
                circles=[circle],
                property_column="PCHEMBL_VALUE",
            )[0]
            == f"{circle[0]}_{circle[1]}_{circle[2]}_{circle[3]}_PCHEMBL_VALUE"
        )


def test_get_faulty_values(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:get_faulty_values.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circles
    """
    assert (
        _get_faulty_entries(
            per_compound_dataframe=per_cpd_dataframe,
            circles=circles,
            property_column="TEST_PCHEMBL_VALUE2",
            include_censored=True,
        )
        == set()
    )
    per_cpd_dataframe.loc["ID1", "TEST_PCHEMBL_VALUE2_Censors"] = "NA"
    assert _get_faulty_entries(
        per_compound_dataframe=per_cpd_dataframe,
        circles=circles,
        property_column="TEST_PCHEMBL_VALUE2",
        include_censored=False,
    ) == {0, 1}


def test_is_same_series(
    circles: list[list[str]],
    per_cpd_dataframe: pd.DataFrame,
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:is_same_series.

    Args:
        circles (list[list[str]]): list of circles
        per_cpd_dataframe (pd.DataFrame): per compound data
    """
    assert _is_same_series(
        per_compound_dataframe=per_cpd_dataframe,
        circle=circles[0],
    )
    assert not _is_same_series(
        per_compound_dataframe=per_cpd_dataframe,
        circle=circles[1],
    )


def test_update_na_per_compound(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:update_na_per_compound.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): solution
    """
    randomized_circle, _ = _randomize_circles(circles=circles)
    lens = [2, 1, 1, 2, 1, 1, 0, 0]
    for circle in randomized_circle:
        nonadditivity = _calculate_nonadditivity(
            values=[
                per_cpd_dataframe.loc[index, "TEST_PCHEMBL_VALUE"] for index in circle
            ],
        )
        _update_na_per_compound(
            per_compound_dataframe=per_cpd_dataframe,
            circle=circle,
            nonadditivity=nonadditivity,
            property_column="TEST_PCHEMBL_VALUE",
            series_column=None,
        )
    for indx, val in enumerate(
        per_cpd_dataframe["TEST_PCHEMBL_VALUE_Nonadditivities"].to_numpy(),
    ):
        assert len(val) == lens[indx]


@pytest.mark.parametrize(
    "id1, idx, result",
    [
        ("ID1", 0, "pure"),
        ("ID1", 1, "mixed"),
        ("ID2", 0, "pure"),
        ("ID3", 0, "pure"),
        ("ID4", 0, "pure"),
        ("ID4", 1, "mixed"),
        ("ID5", 0, "mixed"),
        ("ID6", 0, "mixed"),
    ],
)
def test_update_na_per_compound_series(
    id1: str,
    idx: int,
    result: str,
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:update_na_per_compound_series.

    Args:
        id1 (str): compound id
        idx (tuple[int, int]): index to check
        result (str): expected result
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circles
    """
    randomized_circle, _ = _randomize_circles(circles=circles)
    for circle in randomized_circle:
        nonadditivity = _calculate_nonadditivity(
            values=[per_cpd_dataframe.loc[i, "TEST_PCHEMBL_VALUE"] for i in circle],
        )
        _update_na_per_compound(
            per_compound_dataframe=per_cpd_dataframe,
            circle=circle,
            nonadditivity=nonadditivity,
            property_column="TEST_PCHEMBL_VALUE",
            series_column="ser",
        )
    lens = [2, 1, 1, 2, 1, 1, 0, 0]
    for indx, val in enumerate(
        per_cpd_dataframe["TEST_PCHEMBL_VALUE_Nonadditivities"].to_numpy(),
    ):
        assert len(val) == lens[indx]

    assert (
        per_cpd_dataframe.loc[id1, "TEST_PCHEMBL_VALUE_Nonadditivities"][idx][1]
        == result  # type:ignore
    )
    assert per_cpd_dataframe.loc["ID7", "TEST_PCHEMBL_VALUE_Nonadditivities"] == []


@pytest.mark.parametrize(
    "columns, length, censored, series",
    [
        (["TEST_PCHEMBL_VALUE"], 2, False, None),
        (["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"], 3, False, None),
        (["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"], 4, True, None),
        (["TEST_PCHEMBL_VALUE"], 2, False, "Series"),
    ],
)
def test_calculate_na_output(
    columns: list[str],
    length: int,
    censored: bool,
    series: str | None,
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
    naa_cols: list[str],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:calculate_na_output.

    Args:
        columns (list[str]): property columns
        length (int): expected length of the na dataframe
        censored (bool): whether to include censored data
        series (str | None): series column
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circles
        naa_cols (list[str]): nonadditivity dataframe column names
    """
    na_dataframe, _, _, _ = calculate_na_output(
        circles=circles,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=columns,
        series_column=series,
        include_censored=censored,
    )
    assert len(na_dataframe) == length
    assert np.array_equal(
        na_dataframe.columns.to_numpy(),
        naa_cols,
    )


def test_add_mean_and_std_na_to_df_empty(
    per_cpd_dataframe: pd.DataFrame,
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:add_mean_and_std_na_to_df.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
    """
    data = per_cpd_dataframe[per_cpd_dataframe.Compound_ID.isin(["ID7", "ID8"])].copy()
    na, _, _, _ = calculate_na_output(
        circles=[],
        per_compound_dataframe=data,
        property_columns=["TEST_PCHEMBL_VALUE"],
        series_column=None,
        include_censored=False,
    )
    add_mean_and_std_na_to_df(
        per_compound_dataframe=data,
        property_columns=["TEST_PCHEMBL_VALUE"],
    )
    assert len(na) == 0
    assert np.array_equal(
        data.TEST_PCHEMBL_VALUE_Nonadditivity.to_numpy(),
        np.array([None, None]),
    )


def test_add_mean_and_std_na_to_df(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
    means: list[list[float]],
    stds: list[list[float]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:add_mean_and_std_na_to_df.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circles
        means (list[list[float]]): mean values to expet
        stds (list[list[float]]): std values to expect
    """
    _ = calculate_na_output(
        circles=circles,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=["TEST_PCHEMBL_VALUE"],
        series_column=None,
        include_censored=False,
    )
    add_mean_and_std_na_to_df(
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=["TEST_PCHEMBL_VALUE"],
    )
    lens = [2, 1, 1, 2, 1, 1, 0, 0]
    assert any(
        np.allclose(
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity.to_numpy(),
            s,
            equal_nan=True,
            rtol=0.001,
        )
        for s in means + [[-val for val in mean] for mean in means]
    )
    assert any(
        np.allclose(
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_SD.to_numpy(),
            s,
            equal_nan=True,
            rtol=0.001,
        )
        for s in stds
    )
    assert np.array_equal(
        per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_Count.to_numpy(),
        lens,
        equal_nan=True,
    )


def test_add_mean_and_std_na_to_df_series(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:add_mean_and_std_na_to_df_series.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circes
    """
    _ = calculate_na_output(
        circles=circles,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=["TEST_PCHEMBL_VALUE"],
        series_column="Series",
        include_censored=False,
    )
    add_mean_and_std_na_to_df(
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=["TEST_PCHEMBL_VALUE"],
        series_column="Series",
    )
    na_means_p = [-2.5, 2.5, 2.5, -2.5, np.nan, np.nan, np.nan, np.nan]
    na_means_m = [
        0.8,
        np.nan,
        np.nan,
        -0.8,
        -0.8,
        0.8,
        np.nan,
        np.nan,
    ]

    na_std_p = [0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan]
    na_std_m = [0, np.nan, np.nan, 0, 0, 0, np.nan, np.nan]

    lens_p = [1, 1, 1, 1, 0, 0, 0, 0]
    lens_m = [1, 0, 0, 1, 1, 1, 0, 0]
    assert any(
        np.allclose(
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_Pure.to_numpy(),
            v,
            equal_nan=True,
            rtol=0.001,
        )
        for v in [na_means_p, [-val for val in na_means_p]]
    )
    assert any(
        np.allclose(
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_Mixed.to_numpy(),
            v,
            equal_nan=True,
            rtol=0.001,
        )
        for v in [na_means_m, [-val for val in na_means_m]]
    )
    for colname, truthval in zip(
        ["Pure_SD", "Pure_Count", "Mixed_SD", "Mixed_Count"],
        [na_std_p, lens_p, na_std_m, lens_m],
    ):
        assert np.allclose(
            per_cpd_dataframe[f"TEST_PCHEMBL_VALUE_Nonadditivity_{colname}"].to_numpy(),
            truthval,
            equal_nan=True,
            rtol=0.001,
        )


def test_run_nonadditivty_core(
    per_cpd_dataframe: pd.DataFrame,
    mmp_dataframe: pd.DataFrame,
    means: list[list[float]],
    stds: list[list[float]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:run_nonadditivity_core.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        mmp_dataframe (pd.DataFrame): mmpdb output data
        means (list[list[float]]): mean values to expect
        stds (list[list[float]]): std values to expect
    """
    na_dataframe, circles, circle_ids, _ = run_nonadditivity_core(
        mmp_dataframe=mmp_dataframe,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"],
        no_chiral=False,
        include_censored=False,
        series_column=None,
        verbose=False,
    )
    na_std_v2 = [0, np.nan, np.nan, 0, 0, 0, np.nan, np.nan]
    lens_v2 = [1, 0, 0, 1, 1, 1, 0, 0]
    na_means_v2 = [
        0.8,
        np.nan,
        np.nan,
        -0.8,
        -0.8,
        0.8,
        np.nan,
        np.nan,
    ]

    assert len(na_dataframe) == len(circles) == len(circle_ids) == 3
    lens = [2, 1, 1, 2, 1, 1, 0, 0]
    assert any(
        np.allclose(
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity.to_numpy(),
            s,
            equal_nan=True,
            rtol=0.001,
        )
        for s in means + [[-val for val in mean] for mean in means]
    )
    assert any(
        np.allclose(
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_SD.to_numpy(),
            s,
            equal_nan=True,
            rtol=0.001,
        )
        for s in stds
    )
    assert np.array_equal(
        per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_Count.to_numpy(),
        lens,
        equal_nan=True,
    )
    assert np.allclose(
        per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_Count.to_numpy(),
        lens,
        equal_nan=True,
    )
    assert np.allclose(
        per_cpd_dataframe.TEST_PCHEMBL_VALUE2_Nonadditivity_Count,
        lens_v2,
        equal_nan=True,
    )
    assert np.allclose(
        per_cpd_dataframe.TEST_PCHEMBL_VALUE2_Nonadditivity_SD,
        na_std_v2,
        equal_nan=True,
    )
    assert any(
        np.allclose(
            per_cpd_dataframe.TEST_PCHEMBL_VALUE2_Nonadditivity.to_numpy(),
            s,
            equal_nan=True,
            rtol=0.001,
        )
        for s in [na_means_v2, [-val for val in na_means_v2]]
    )


def test_run_nonadditivty_core_censored(
    per_cpd_dataframe: pd.DataFrame,
    mmp_dataframe: pd.DataFrame,
    means: list[list[float]],
    stds: list[list[float]],
) -> None:
    """Test nonadditivity.workflows.nonadditivity_core:run_nonadditivity_core.

    This time including censored values.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        mmp_dataframe (pd.DataFrame): mmpdb output data
        means (list[list[float]]): mean values to expect
        stds (list[list[float]]): std values to expect.
    """
    na_dataframe, circles, circle_ids, _ = run_nonadditivity_core(
        mmp_dataframe=mmp_dataframe,
        per_compound_dataframe=per_cpd_dataframe,
        property_columns=["TEST_PCHEMBL_VALUE", "TEST_PCHEMBL_VALUE2"],
        no_chiral=False,
        include_censored=True,
        series_column=None,
        verbose=False,
    )

    assert len(na_dataframe) == len(circles) == len(circle_ids) == 4
    lens = [2, 1, 1, 2, 1, 1, 0, 0]
    for nam, nasd, nac in zip(
        [
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity.to_numpy(),
            per_cpd_dataframe.TEST_PCHEMBL_VALUE2_Nonadditivity.to_numpy(),
        ],
        [
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_SD.to_numpy(),
            per_cpd_dataframe.TEST_PCHEMBL_VALUE2_Nonadditivity_SD.to_numpy(),
        ],
        [
            per_cpd_dataframe.TEST_PCHEMBL_VALUE_Nonadditivity_Count.to_numpy(),
            per_cpd_dataframe.TEST_PCHEMBL_VALUE2_Nonadditivity_Count.to_numpy(),
        ],
    ):
        assert any(
            np.allclose(
                nam,
                m,
                equal_nan=True,
                rtol=0.001,
            )
            for m in means + [[-v for v in mean] for mean in means]
        )
        assert any(
            np.allclose(
                nasd,
                s,
                equal_nan=True,
                rtol=0.001,
            )
            for s in stds
        )

        assert np.allclose(
            nac,
            lens,
            equal_nan=True,
        )
