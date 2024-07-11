"""Test nonaddicitivity.workfow.canonicalize_output."""

import pandas as pd
import pytest
from rdkit import Chem

from nonadditivity.classification import Circle, classify
from nonadditivity.workflow.canonicalize_output import (
    _get_correct_order,
    _get_correct_transformation_smarts,
    _get_props,
    _get_reverse,
    canonicalize_na_dataframe,
    get_transformation_df,
)
from nonadditivity.workflow.nonadditivity_core import calculate_na_output


@pytest.mark.parametrize(
    "inval, sol",
    [("CC>>CCC", "CCC>>CC"), ("CADFN>ADFCNE>>SD", "SD>>CADFN>ADFCNE")],
)
def test_get_reverse(inval: str, sol: str) -> None:
    """Test nonaddicitivity.workfow.canonicalize_output:_get_reverse.

    Args:
        inval (str): test input
        sol (str): expected output
    """
    with pytest.raises(ValueError):
        _get_reverse("asdf")
    assert _get_reverse(inval) == sol


@pytest.mark.parametrize(
    "transformations, reverse, sol",
    [
        (("CC>>CCC", "AA>>AAA"), (True, True), ("CCC>>CC", "AAA>>AA")),
        (("CC>>CCC", "AA>>AAA"), (True, False), ("CCC>>CC", "AA>>AAA")),
        (("CC>>CCC", "AA>>AAA"), (False, True), ("CC>>CCC", "AAA>>AA")),
        (("CC>>CCC", "AA>>AAA"), (False, False), ("CC>>CCC", "AA>>AAA")),
    ],
)
def test_get_correct_transformation_smarts(
    transformations: tuple[str, str],
    reverse: tuple[bool, bool],
    sol: str,
) -> None:
    """Test ...:_get_correct_transformation_smarts.

    Args:
        transformations (tuple[str, str]): test input transformations
        reverse (tuple[bool, bool]): test input reversed values
        sol (str): expected output
    """
    assert _get_correct_transformation_smarts(*transformations, *reverse) == sol


@pytest.mark.parametrize(
    "reverse, sol",
    [
        ((True, True), (3, 4, 1, 2)),
        ((True, False), (2, 1, 4, 3)),
        ((False, True), (4, 3, 2, 1)),
        ((False, False), (1, 2, 3, 4)),
    ],
)
def test_get_correct_order(reverse: tuple[bool, bool], sol: str) -> None:
    """Test nonaddicitivity.workfow.canonicalize_output:_get_correct_order.

    Args:
        reverse (tuple[bool, bool]): reverse input
        sol (str): expectec output
    """
    assert _get_correct_order(*reverse) == sol


def test_get_props() -> None:
    """Test nonaddicitivity.workfow.canonicalize_output:_get_props."""
    solution = list(Circle.classification_keys.values())
    for rem in (
        "transformation_at_ortho",
        "ortho_substituent_introduced",
        "num_atoms_between_r_groups",
    ):
        solution.remove(rem)
    solution = ["num_atoms_between_r_groups", "ortho_classification", *solution]
    assert _get_props() == solution


def test_canonicalize_na_dataframe(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonaddicitivity.workfow.canonicalize_output:canonicalize_na_dataframe.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circles
    """
    nadf, _, _, _ = calculate_na_output(
        circles,
        per_cpd_dataframe,
        ["TEST_PCHEMBL_VALUE"],
        include_censored=False,
    )
    cnadf = canonicalize_na_dataframe(na_dataframe=nadf)
    canonical_tr_df = get_transformation_df(cnadf)
    nadf_cols = list(nadf.columns.values)
    tr_df_sol = sorted(["Property", "Transformation", "Val1", "Val2", "Series"])
    assert sorted(nadf_cols) == sorted(cnadf.columns.values)
    assert sorted(canonical_tr_df) == tr_df_sol


def test_canonicalize_na_dataframe_w_classify(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
) -> None:
    """Test nonaddicitivity.workfow.canonicalize_output:canonicalize_na_dataframe.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compond data
        circles (list[list[str]]): list of circles
    """
    per_cpd_dataframe["RDKit_Molecules"] = [
        Chem.MolFromSmiles(sm) for sm in per_cpd_dataframe.SMILES.to_numpy()
    ]
    nadf, _, _, _ = calculate_na_output(
        circles,
        per_cpd_dataframe,
        ["TEST_PCHEMBL_VALUE"],
        include_censored=False,
    )
    _ = classify(per_cpd_dataframe, nadf)
    cnadf = canonicalize_na_dataframe(na_dataframe=nadf)
    canonical_tr_df = get_transformation_df(cnadf)
    nadf_cols = list(nadf.columns.values)
    tr_df_sol = sorted(
        ["Property", "Transformation", "Val1", "Val2", "Series", *_get_props()],
    )
    assert sorted(nadf_cols) == sorted(cnadf.columns.values)
    assert sorted(canonical_tr_df) == tr_df_sol
