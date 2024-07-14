"""Implement classification on all three levels.

Classification submodule of the NonadditivityAnalysis module. All Functions and objects
used here are higly specific. Don't use them on their own.
"""

import pandas as pd
from tqdm.auto import tqdm

from nonadditivity.classification import Circle, Compound, Transformation


def _create_compound_dict(
    per_compound_dataframe: pd.DataFrame,
) -> dict[str, Compound]:
    """Create dict mapping id to compound object.

    Creates dict mapping str to Compound objects for every entry in
    per_compound_dataframe.

    Args:
        per_compound_dataframe (pd.DataFrame): Dataframe w/ 'RDKit_Moleculs' & 'SMILES'

    Returns:
        dict[str, Compound]: dict mapping str to compound objects
    """
    return {
        i: Compound(molecule=m, compound_id=i, smiles=s)
        for m, i, s in zip(
            per_compound_dataframe.RDKit_Molecules.to_numpy(),
            per_compound_dataframe.index.to_numpy(),
            per_compound_dataframe.SMILES.to_numpy(),
        )
    }


def _update_per_compound_dataframe(
    per_compound_dataframe: pd.DataFrame,
    compound_dict: dict[str, Compound],
) -> None:
    """Add classification details to per compound dataframe.

    Adding 'num_stereocenters' and 'num_ortho_configurations' and
    'has_unassigned_stereocenters' for every compound to the per compound dataframe.

    Args:
        per_compound_dataframe (pd.DataFrame): dataframe to add info to
        compound_dict (dict[str, Compound]): dictionary containing all compounds
        present in per_compound_dataframe.
    """
    props = Compound.Properties

    for compound_property, compound_property_str in (
        pbar := tqdm(
            Compound.classification_keys.items(),
            total=len(Compound.classification_keys.items()),
            desc="Calculating Compound properties",
            ascii="░▒█",
        )
    ):
        if compound_property in (
            props.NUM_HEAVY_ATOMS,
            props.MORGANFP,
            props.ORTHO_INDICES,
            props.NUM_ORTHO_CONFIGURATIONS,
        ):
            continue
        pbar.set_description(
            desc=(
                "Calculating Compound properties: "
                f"{compound_property_str.replace('_',' ')}"
            ),
        )
        values = []
        for key in tqdm(per_compound_dataframe.index.to_numpy()):
            values.append(compound_dict[key].get_property(compound_property))
        per_compound_dataframe[compound_property_str] = values


def classify_compounds(
    per_compound_dataframe: pd.DataFrame,
) -> dict[str, Compound]:
    """Classify all compounds in per compound dataframe and create compound instances.

    Creates Compound objects for every entry in the per_compound_dataframe and adds
    columns "num_stereocenters" and "num_ortho_patterns to the per_compound_dataframe.

    Args:
        per_compound_dataframe (pd.DataFrame): df containing compounds

    Returns:
        dict[str, Compound]: dict mapping compound id to compound object
    """
    compound_dict = _create_compound_dict(
        per_compound_dataframe=per_compound_dataframe,
    )
    _update_per_compound_dataframe(
        per_compound_dataframe=per_compound_dataframe,
        compound_dict=compound_dict,
    )
    return compound_dict


def _get_na_compounds(
    na_dataframe: pd.DataFrame,
    compound_dict: dict[str, Compound],
) -> list[list[Compound]]:
    """Get a list of compounds for every row in the naa dataframe.

    Returns list lists of compounds that are present in the nonadditivity circles
    in the na_dataframe.

    Args:
        na_dataframe (pd.DataFrame): dataframe containint nonadditivity circles
        compound_dict (dict[str, Compound]): dict mapping compound id to compound object

    Returns:
        list[list[Compound]]: list of lists with compounds in na_dataframe
    """
    return [
        [compound_dict[i] for i in ids]
        for ids in zip(
            na_dataframe.Compound1.to_numpy(),
            na_dataframe.Compound2.to_numpy(),
            na_dataframe.Compound3.to_numpy(),
            na_dataframe.Compound4.to_numpy(),
        )
    ]


def _get_transformations_for_na_dataframe(
    na_compounds: list[list[Compound]],
    per_compound_dataframe: pd.DataFrame,
) -> list[tuple[Transformation, ...]]:
    """Get transformation objects for transformations in na dataframe.

    Returns a list of tuples of transformation, so a Transformation object for
    every transformation in a circle of the na_dataframe is generated:

    CPD1 ======= T1 ====== > CPD2
      |                        |
      |                        |
     T2                       T3
      |                        |
      /                        /
    CPD4 ======= T4 ====== > CPD3

    Args:
        na_compounds (list[list[Compound]]): list of lists for compounds in\
            naa dataframe circles
        per_compound_dataframe (pd.DataFrame): per compound dataframe

    Returns:
        list[tuple[Transformation, ...]]:\
            list of tuples containing transformations for all the naa circles.
    """
    return [
        tuple(
            Transformation(
                compound_1=c[i],
                compound_2=c[j],
                constant_smarts=per_compound_dataframe.loc[
                    c[i].compound_id,
                    "Neighbor_dict",
                ][c[j].compound_id][1],
                transformation_smarts=per_compound_dataframe.loc[
                    c[i].compound_id,
                    "Neighbor_dict",
                ][c[j].compound_id][0],
            )
            for i, j in ((0, 1), (0, 3), (1, 2), (3, 2))
        )
        for c in tqdm(
            na_compounds,
            total=len(na_compounds),
            desc="Classifying Transformation",
            ascii="░▒█",
        )
    ]


def _create_circle_objects(
    na_transformations: list[tuple[Transformation, ...]],
) -> list[Circle]:
    """Return list of Circle objects.

    Args:
        na_transformations (list[ tuple[Transformation, ...] ]): \
            list of tuples where every\
                tuple has the four transformations makeing up a circle.

    Returns:
        list[Circle]: Circle objects.
    """
    return [
        Circle(
            transformation_1=transformation_1,
            transformation_2=transformation_2,
            transformation_3=transformation_3,
            transformation_4=transformation_4,
        )
        for (
            transformation_1,
            transformation_2,
            transformation_3,
            transformation_4,
        ) in na_transformations
    ]


def _classify_ortho(circle: Circle) -> str:
    """Create ortho classification string.

    Melts the two classifications HAS_ORTHO_SUBSTITUENT_INTRODUCED
    and HAS_TRANSFORMATION_AT_ORTHO into one so ortho_classification
    can either be 'Both', 'Introduced', 'Changed' or 'None'.

    Args:
        circle (Circle): Circle to classify

    Returns:
        str: 'None', 'Both', 'Introduced' or 'Changed'.
    """
    if circle.get_property(
        circle.Properties.HAS_ORTHO_SUBSTITUENT_INTRODUCED,
    ) and circle.get_property(circle.Properties.HAS_TRANSFORMATION_AT_ORTHO):
        return "Both"
    if circle.get_property(circle.Properties.HAS_ORTHO_SUBSTITUENT_INTRODUCED):
        return "Introduced"
    if circle.get_property(circle.Properties.HAS_TRANSFORMATION_AT_ORTHO):
        return "Changed"
    return "None"


def _classify_circles(
    circles: list[Circle],
    property_key: Circle.Properties,
) -> list[str] | list[float] | list[tuple[int]] | list[bool]:
    """Gets the classification for <property_key> for a list of circles.

    Args:
        circles (list[Circle]): circles to classify.
        property_key (Circle.Properties): property to get.

    Returns:
        list[str] | list[float] | list[tuple[int]] | list[bool]:
        classification for circles and property_key
    """
    classification: list[str] | list[float] | list[tuple[int]] = []  # type:ignore
    for circle in circles:
        classification.append(circle.get_property(property_key))  # type: ignore
    return classification


def _is_surprisingly_nonadditive(circle: Circle) -> bool:
    """Check whether a circle's nonadditivity is surprising.

    Use keyword "circle" for this to work.

    A circle is considered surprising i fnone of the following is true:
    Distance between R groups ≤ 2 atoms
    Tanimoto similarity of the transformation < 0.4
    Number of exchanged heavy atoms > 10
    Linker exchange transformations
    Transformations with unassigned or inverted stereocenters

    Args:
        circle (Circle): circle to check.

    Returns:
        bool: True if surprising false if Mundane.
    """
    if circle.get_property(Circle.Properties.DISTANCE_BETWEEN_R_GROUPS) <= 2:
        return False
    if circle.get_property(Circle.Properties.MIN_TANIMOTO) < 0.4:
        return False
    if circle.get_property(Circle.Properties.NUM_HEAVY_ATOMS_DIFF) > 10:
        return False
    if circle.get_property(Circle.Properties.MAX_NUM_MMPDB_CUTS) > 1:
        return False
    return (
        circle.get_property(Circle.Properties.HAS_INVERSION_IN_TRANSFORMATION) == "None"
    )


def _update_na_dataframe(
    na_dataframe: pd.DataFrame,
    circles: list[Circle],
) -> None:
    """Add circle classification columns to the na_dataframe.

    Args:
        na_dataframe (pd.DataFrame): dataframe to add columns to.
        circles (list[Circle]): list of circle objects in the order of them occuring\
            in the na_dataframe.
    """
    props = Circle.Properties
    for circle_property, circle_property_str in (
        pbar := tqdm(
            Circle.classification_keys.items(),
            total=len(Circle.classification_keys.items()),
            desc="Calculating Circle properties",
            ascii="░▒█",
        )
    ):
        if circle_property in (
            props.DISTANCE_BETWEEN_R_GROUPS,
            props.HAS_ORTHO_SUBSTITUENT_INTRODUCED,
            props.HAS_TRANSFORMATION_AT_ORTHO,
        ):
            continue
        pbar.set_description(
            desc=(
                "Calculating Circle properties: "
                f"{circle_property_str.replace('_',' ')}"
            ),
        )
        na_dataframe[circle_property_str] = _classify_circles(
            circles=circles,
            property_key=circle_property,
        )

    na_dataframe["ortho_classification"] = [_classify_ortho(c) for c in circles]
    na_dataframe["num_atoms_between_r_groups"] = _classify_circles(
        circles=circles,
        property_key=props.DISTANCE_BETWEEN_R_GROUPS,
    )
    na_dataframe["classification"] = [
        "surprising" if _is_surprisingly_nonadditive(c) else "mundane" for c in circles
    ]


def classify_circles(
    na_transformations: list[tuple[Transformation, ...]],
    na_dataframe: pd.DataFrame,
) -> list[Circle]:
    """Create circle objects and classify them.

    Creates Circle objects for every entry in the na_dataframe and classifies
    them. Then adds the classification to the na_dataframe.

    Args:
        na_transformations (list[ tuple[Transformation, ...] ]):
            na_transformations the circles in na_dataframe are made of.
        na_dataframe (pd.DataFrame): na_dataframe with circles to classify.

    Returns:
        list[Circle]: list of classified circle objects.
    """
    circles = _create_circle_objects(na_transformations=na_transformations)
    _update_na_dataframe(na_dataframe=na_dataframe, circles=circles)
    return circles


def classify(
    per_compound_dataframe: pd.DataFrame,
    na_dataframe: pd.DataFrame,
) -> tuple[list[list[Compound]], list[tuple[Transformation, ...]], list[Circle]]:
    """Classify compounds, circles and transformations in the circle.

    Classifies all compound in the per_compound_dataframe and then classifies all
    the circles in the na_dataframe.

    Args:
        per_compound_dataframe (pd.DataFrame): dataframe containing compounds found\
            in the nonadditivity circles.
        na_dataframe (pd.DataFrame): dataframe containing circles to be classified.

    Returns:
        tuple[ list[list[Compound]], list[tuple[Transformation, ...]], list[Circle]]:\
             tuple containing\
            list of lists of compounds found in the nonadditivity circles,\
            list of tuples of transformations found in the nonadditivty circles\
            and list of circles in the na_dataframe.
    """
    compound_dict = classify_compounds(
        per_compound_dataframe=per_compound_dataframe,
    )
    na_compounds = _get_na_compounds(
        na_dataframe=na_dataframe,
        compound_dict=compound_dict,
    )
    na_transformations = _get_transformations_for_na_dataframe(
        na_compounds=na_compounds,
        per_compound_dataframe=per_compound_dataframe,
    )
    na_circles = classify_circles(
        na_transformations=na_transformations,
        na_dataframe=na_dataframe,
    )
    return na_compounds, na_transformations, na_circles
