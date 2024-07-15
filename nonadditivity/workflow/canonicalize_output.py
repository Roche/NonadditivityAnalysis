"""Canonicalizing nonadditivity output.

Helper Module that canonicalizes output files in a way,
that there are no more transformations in the output,
that are the reverse of another transformation.
"""

import contextlib

import pandas as pd

from nonadditivity.classification import Circle
from nonadditivity.utils.math import _calculate_nonadditivity
from nonadditivity.workflow.nonadditivity_core import _create_circle_ids


def _get_reverse(transformation: str) -> str:
    """Get reverse transformation: 'A>>B' is transformed to 'B>>A'.

    Args:
        transformation (str): transformation smarts

    Returns:
        str: reversed transformation smarts

    Raises:
        ValueError: Raised if '>>' not in transformation.
    """
    if ">>" not in transformation:
        raise ValueError("transformation has to contain '>>'")
    parts = transformation.split(">>")
    return parts[1] + ">>" + parts[0]


def _get_correct_transformation_smarts(
    t1: str,
    t2: str,
    t1_reversed: bool,
    t2_reversed: bool,
) -> tuple[str, ...]:
    """Get the correct version of t1 and t2.

    reverses them if t1_changed = True, resp. for t2.

    Args:
        t1 (str): transformation 1
        t2 (str): transformation 2
        t1_reversed (bool): whether transformation 1 should be reversed.
        t2_reversed (bool): whether transformation 2 should be reversed.

    Returns:
        tuple[str, ...]: correct transformations.
    """
    return (
        _get_reverse(t1) if t1_reversed else t1,
        _get_reverse(t2) if t2_reversed else t2,
    )


def _get_correct_order(
    t1_reversed: bool,
    t2_reversed: bool,
) -> tuple[int, ...]:
    """Get correct index order for canonical transformation.

    Transforms the indices (1,2,3,4) according to wheter any or all of the
    transformations are reversed.

    Args:
        t1_reversed (bool): transformation1 is reversed
        t2_reversed (bool): transformation2 is reversed

    Returns:
        tuple[int, ...]: new order of the circle
    """
    if all([t1_reversed, t2_reversed]):
        return (3, 4, 1, 2)
    if t1_reversed:
        return (2, 1, 4, 3)
    if t2_reversed:
        return (4, 3, 2, 1)
    return (1, 2, 3, 4)


def _get_props() -> list[str]:
    """Return circle porperty names.

    Returns all Properties in nonadditivity.classification_classes.Circle
    that are not ortho_classification or num_bonds_between_atoms.

    Returns:
        list[str]: list of descripytion keys
    """
    return (
        ["num_atoms_between_r_groups", "ortho_classification"]
        + [
            val
            for val in Circle.classification_keys.values()
            if "ortho" not in val and "between" not in val
        ]
        + ["classification"]
    )


def _add_to_dict(
    row: pd.Series,
    index: int,
    df_dict: dict[str, list | pd.Series],
    t1_reversed: bool,
    t2_reversed: bool,
) -> None:
    """If necessary reverse transformations and append the new values to df_dict.

    Args:
        row (pd.Series): row to be transformed
        index (int): index of this row in dataframe
        df_dict (dict[str, list]): dict to add values to
        t1_reversed (bool): whether t1 has to be reversed
        t2_reversed (bool): whether t2 has to be reversed
    """
    order = (1, 2, 3, 4)
    transformation1, transformation2 = (
        row["Transformation1"],
        row["Transformation2"],
    )
    if any([t1_reversed, t2_reversed]):
        new_order = _get_correct_order(t1_reversed, t2_reversed)
        transformation1, transformation2 = _get_correct_transformation_smarts(
            transformation1,
            transformation2,
            t1_reversed,
            t2_reversed,
        )
    else:
        new_order = order
    for entry, new in zip(order, new_order):
        for name in ("Compound", "SMILES", "Prop_Cpd"):
            df_dict[f"{name}{entry}"].append(row[f"{name}{new}"])

    df_dict["Property"].append(row["Property"])
    with contextlib.suppress(KeyError):
        df_dict["Series"].append(row["Series"])

    df_dict["Circle_ID"].append(
        _create_circle_ids(
            [[df_dict[f"Compound{i}"][index] for i in range(1, 5)]],
            row["Property"],
        )[0],
    )
    old_na = row["Nonadditivity"]
    new_na = _calculate_nonadditivity(
        values=[row[f"Prop_Cpd{i}"] for i in new_order],
    )
    df_dict["Nonadditivity"].append(new_na)
    theo_sign = 1 if old_na == new_na else -1
    df_dict["Theo_Quantile"].append(row["Theo_Quantile"] * theo_sign)
    df_dict["Transformation1"].append(transformation1)
    df_dict["Transformation2"].append(transformation2)

    for prop in _get_props():
        try:
            df_dict[prop].append(row[prop])
        except KeyError:
            continue


def _create_df_dict(
    na_dataframe: pd.DataFrame,
) -> dict[str, list | pd.Series]:
    """Create dict with unique transformations.

    Creates a dictionary containing all na_dataframes values but with only unique
    transformations.

    Args:
        na_dataframe (pd.DataFrame): dataframe to be canonicalized

    Returns:
        dict[str, list | pd.Series]: dict[columns, canonicalized values]
    """
    df_dict: dict[str, list | pd.Series] = {
        str(key): [] for key in na_dataframe.columns.to_numpy()
    }
    seen = set()
    for index, (_, row) in enumerate(na_dataframe.iterrows()):
        t1_changed, t2_changed = False, False
        t1, t2 = row["Transformation1"], row["Transformation2"]
        if t1 not in seen:
            if _get_reverse(t1) in seen:
                t1_changed = True
            else:
                seen.add(t1)
        if t2 not in seen:
            if _get_reverse(t2) in seen:
                t2_changed = True
            else:
                seen.add(t2)
        _add_to_dict(row, index, df_dict, t1_changed, t2_changed)
    return df_dict


def canonicalize_na_dataframe(
    na_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Canonicalize nonadditivity dataframe.

    Creates a dataframe that contains only canonicalized transformations,
    i.e. only one of the following is present: 'Cl>>F', 'F>>Cl'

    Args:
        na_dataframe (pd.DataFrame): dataframe to be transformed

    Returns:
        pd.DataFrame: canonicalized dataframe.
    """
    canonical_df = pd.DataFrame()
    try:
        na_dataframe_fixed = na_dataframe.copy().drop(
            columns=[
                "has_transformation_at_ortho",
                "has_ortho_substituent_introduced",
            ],
        )
    except KeyError:
        na_dataframe_fixed = na_dataframe
    df_dict = _create_df_dict(na_dataframe_fixed)
    for col in na_dataframe_fixed.columns.to_numpy():
        if col not in df_dict:
            continue
        canonical_df[col] = df_dict[col]
    return canonical_df


def get_transformation_df(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Create dataframe with canonical trasnformations.

    Creates a dataframe containing all the transformations and the respective
    activity values.

    Args:
        canonical_df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: df containing 'Property', 'Transformation', 'Val1', 'Val2'.
    """
    transformations_dataframe = pd.DataFrame()
    transformations = []
    val1, val2 = [], []
    targets = []

    for tr1, tr2, prop1, prop2, prop3, prop4, target in zip(
        canonical_df.Transformation1.to_numpy(),
        canonical_df.Transformation2.to_numpy(),
        canonical_df.Prop_Cpd1.to_numpy(),
        canonical_df.Prop_Cpd2.to_numpy(),
        canonical_df.Prop_Cpd3.to_numpy(),
        canonical_df.Prop_Cpd4.to_numpy(),
        canonical_df.Property.to_numpy(),
    ):
        for _ in range(4):
            targets.append(target)
        transformations.append(tr1)
        val1.append(prop1)
        val2.append(prop2)
        transformations.append(tr1)
        val1.append(prop4)
        val2.append(prop3)
        transformations.append(tr2)
        val1.append(prop1)
        val2.append(prop4)
        transformations.append(tr2)
        val1.append(prop2)
        val2.append(prop3)
    for props in _get_props():
        try:
            transformations_dataframe[props] = [
                canonical_df.loc[index, props]
                for index in canonical_df.index.to_numpy()
                for _ in range(4)
            ]
        except KeyError:
            continue
    transformations_dataframe["Property"] = targets
    transformations_dataframe["Transformation"] = transformations
    transformations_dataframe["Val1"] = val1
    transformations_dataframe["Val2"] = val2
    with contextlib.suppress(KeyError):
        transformations_dataframe["Series"] = [
            series for series in canonical_df["Series"].to_numpy() for _ in range(4)
        ]
    return transformations_dataframe
