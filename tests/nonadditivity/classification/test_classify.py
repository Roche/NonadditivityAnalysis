"""Test nonadditivity.classification.classify."""

import pandas as pd
import pytest
from rdkit import Chem

from nonadditivity.classification import Circle, Compound, Transformation
from nonadditivity.classification.classify import (
    _classify_circles,
    _classify_ortho,
    _create_circle_objects,
    _create_compound_dict,
    _get_na_compounds,
    _get_transformations_for_na_dataframe,
    _is_surprisingly_nonadditive,
    _update_na_dataframe,
    _update_per_compound_dataframe,
    classify,
    classify_circles,
    classify_compounds,
)
from nonadditivity.workflow.nonadditivity_core import calculate_na_output

Props = Circle.Properties
Solutions = [
    (Props.H_BOND_DONORS_DIFF, 1),
    (Props.H_BOND_ACCEPTORS_DIFF, 1),
    (Props.FORMAL_CHARGE_DIFF, 0),
    (Props.POLAR_SURFACE_AREA_DIFF, 20.230000000000004),
    (Props.NUM_ROTATABLE_BONDS_DIFF, 2),
    (Props.SP3_CARBON_DIFF, 3),
    (Props.LOG_P_DIFF, 0.3460000000000001),
    (Props.NUM_HEAVY_ATOMS_DIFF, 4),
    (Props.CHI0_DIFF, 3.1547005383792524),
    (Props.CHI1_DIFF, 1.8045304526403108),
    (Props.CHI2_DIFF, 1.3427031358353583),
    (Props.HAS_TERTIARY_AMIDE_FORMED, "False"),
    (Props.HAS_INVERSION_IN_TRANSFORMATION, "None"),
    (Props.MAX_NUM_MMPDB_CUTS, 1),
    (Props.MAX_HEAVY_ATOM_IN_TRANSFORMATION, 4),
    (Props.COMPOUND_STEREO_CLASSIFICATION, "Assigned"),
    (Props.MIN_TANIMOTO, 0.6626506024096386),
    (Props.SUBSTITUENT_ON_SAME_RING_SYSYTEM, False),
]


@pytest.fixture()
def cpd_dict(per_cpd_dataframe: pd.DataFrame) -> dict[str, Compound]:
    """Create dict mapping id -> Compound.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data

    Returns:
        dict[str, Compound]: id -> Compound
    """
    per_cpd_dataframe["RDKit_Molecules"] = [
        Chem.MolFromSmiles(sm)
        for sm in per_cpd_dataframe.SMILES.to_numpy()  # type: ignore pylint:disable=E1101
    ]
    return _create_compound_dict(per_compound_dataframe=per_cpd_dataframe)


def test_create_compound_dict(per_cpd_dataframe: pd.DataFrame) -> None:
    """Test nonadditivity.classification.classify:_create_compound_dict.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
    """
    per_cpd_dataframe["RDKit_Molecules"] = [
        Chem.MolFromSmiles(sm)
        for sm in per_cpd_dataframe.SMILES.to_numpy()  # type: ignore pylint:disable=E1101
    ]
    cpd = _create_compound_dict(per_compound_dataframe=per_cpd_dataframe)
    for key, value in cpd.items():
        assert per_cpd_dataframe.loc[key, "SMILES"] == value.smiles
        assert key == value.compound_id
        assert Chem.MolToSmiles(  # type: ignore pylint:disable=E1101
            per_cpd_dataframe.loc[key, "RDKit_Molecules"],
        ) == Chem.MolToSmiles(  # type: ignore pylint:disable=E1101
            value.rdkit_molecule,
        )


@pytest.mark.parametrize(
    "column, solution",
    [
        ("num_stereocenters", [0 for _ in range(8)]),
        ("has_unassigned_stereocenters", [False for _ in range(8)]),
    ],
)
def test_update_per_compound_dataframe(
    per_cpd_dataframe: pd.DataFrame,
    cpd_dict: dict[str, Compound],
    column: str,
    solution: list[int] | list[bool],
) -> None:
    """Test nonadditivity.classification.classify:_update_per_compound_dataframe.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        cpd_dict (dict[str, Compound]): dict mapping str -> Compound
        column (str): classification column name
        solution (list[int] | list[bool]): expected solution.
    """
    _update_per_compound_dataframe(
        per_compound_dataframe=per_cpd_dataframe,
        compound_dict=cpd_dict,
    )
    assert list(per_cpd_dataframe[column].values) == solution


@pytest.mark.parametrize(
    "column, solution",
    [
        ("num_stereocenters", [0 for _ in range(8)]),
        ("has_unassigned_stereocenters", [False for _ in range(8)]),
    ],
)
def test_classify_compounds(
    per_cpd_dataframe: pd.DataFrame,
    column: str,
    solution: list[int] | list[bool],
) -> None:
    """Test nonadditivity.classification.classify:classify_compounds.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        column (str): classification column
        solution (list[int] | list[bool]): expecte doutput.
    """
    per_cpd_dataframe["RDKit_Molecules"] = [
        Chem.MolFromSmiles(sm) for sm in per_cpd_dataframe.SMILES.to_numpy()
    ]
    classify_compounds(per_cpd_dataframe)
    assert list(per_cpd_dataframe[column].to_numpy()) == solution


def test_get_na_compounds(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
    cpd_dict: dict[str, Compound],
) -> None:
    """Test nonadditivity.classification.classify:_get_na_compounds.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (list[list[str]]): list of circles
        cpd_dict (dict[str, Compound]): dict mapping str->Compound
    """
    per_cpd_dataframe["TEST_PCHEMBL_VALUE_Censors"] = ["" for _ in range(8)]
    na_df, _, _, _ = calculate_na_output(
        per_compound_dataframe=per_cpd_dataframe,
        circles=circles,
        include_censored=False,
        property_columns=["TEST_PCHEMBL_VALUE"],
    )
    na_compounds = _get_na_compounds(na_df, cpd_dict)
    for index, cmpds in enumerate(na_compounds):
        for i in range(4):
            assert cmpds[i].compound_id == na_df.loc[index, f"Compound{i+1}"]
            assert cmpds[i].smiles == na_df.loc[index, f"SMILES{i+1}"]


def test_get_transformations_for_na_dataframe(
    per_cpd_dataframe: pd.DataFrame,
    cpd_dict: dict[str, Compound],
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.classification.classify:_get_transformations_for_na_dataframe.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        cpd_dict (dict[str, Compound]): dict mapping str->Compound
        circles (list[list[str]]): list of circles
    """
    per_cpd_dataframe["TEST_PCHEMBL_VALUE_Censors"] = ["" for _ in range(8)]
    na_df, rcircles, _, _ = calculate_na_output(
        per_compound_dataframe=per_cpd_dataframe,
        circles=circles,
        include_censored=False,
        property_columns=["TEST_PCHEMBL_VALUE"],
    )
    na_compounds = _get_na_compounds(na_df, cpd_dict)
    na_transformations = _get_transformations_for_na_dataframe(
        na_compounds=na_compounds,
        per_compound_dataframe=per_cpd_dataframe,
    )
    for index, transformations in enumerate(na_transformations):
        for transformation, cindex1, cindex2 in zip(
            transformations,
            (0, 0, 1, 3),
            (1, 3, 2, 2),
        ):
            assert transformation.compound_1.compound_id == rcircles[index][cindex1]
            assert transformation.compound_2.compound_id == rcircles[index][cindex2]


def test_create_circle_objects(
    per_cpd_dataframe: pd.DataFrame,
    cpd_dict: dict[str, Compound],
    circles: list[list[str]],
) -> None:
    """Test nonadditivity.classification.classify:_create_circle_objects.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        cpd_dict (dict[str, Compound]): dict mapping str->Compound
        circles (list[list[str]]): list of circles
    """
    per_cpd_dataframe["TEST_PCHEMBL_VALUE_Censors"] = ["" for _ in range(8)]
    na_df, _, _, _ = calculate_na_output(
        per_compound_dataframe=per_cpd_dataframe,
        circles=circles,
        include_censored=False,
        property_columns=["TEST_PCHEMBL_VALUE"],
    )
    na_compounds = _get_na_compounds(na_df, cpd_dict)
    na_transformations = _get_transformations_for_na_dataframe(
        na_compounds=na_compounds,
        per_compound_dataframe=per_cpd_dataframe,
    )
    na_circles = _create_circle_objects(na_transformations=na_transformations)
    for index, circle in enumerate(na_circles):
        assert circle.transformation_1 == na_transformations[index][0]
        assert circle.transformation_2 == na_transformations[index][1]
        assert circle.transformation_3 == na_transformations[index][2]
        assert circle.transformation_4 == na_transformations[index][3]


@pytest.mark.parametrize(
    "circ, solution",
    [
        ("ortho_none_circle", "None"),
        ("circle", "Introduced"),
        ("ortho_both_circle", "Both"),
        ("ortho_exchanged_circle", "Changed"),
    ],
)
def test_classify_ortho(
    circ: str,
    solution: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test nonadditivity.classification.classify:_classify_ortho.

    Args:
        circ (str): circlefixture name
        solution (str): expected value
        request (pytest.FixtureRequest): pytest magic
    """
    assert _classify_ortho(request.getfixturevalue(circ)) == solution


@pytest.mark.parametrize(
    "prop, solution",
    [
        (Props.DISTANCE_BETWEEN_R_GROUPS, 5),
        (Props.HAS_TRANSFORMATION_AT_ORTHO, False),
        (Props.HAS_ORTHO_SUBSTITUENT_INTRODUCED, True),
        *Solutions,
    ],
)
def test_classify_circle_special(
    circle: Circle,
    prop: Props,
    solution: float | str | bool,
) -> None:
    """Test nonadditivity.classification.classify:_classify_circles.

    Args:
        circle (Circle): circle object
        prop (Props): property to get
        solution (float | str | bool): expected output.
    """
    result = _classify_circles([circle], prop)
    if isinstance(prop, str):
        index = int("atoms" in prop)
        assert result[index] == result
    assert result == [solution]


@pytest.mark.parametrize(
    "fixture_name, solution",
    [
        ("circle", True),
        ("circle_2", True),
        ("ortho_none_circle", False),
        ("ortho_exchanged_circle", True),
        ("ortho_both_circle", False),
    ],
)
def test_is_surprisingly_nonadditive(
    fixture_name: str,
    solution: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test nonadditivity.classification.classify:_is_surprisingly_nonadditive.

    Args:
        fixture_name (str): fixture name
        solution (str): expected output
        request (pytest.FixtureRequest): pytest magic
    """
    circle: Circle = request.getfixturevalue(fixture_name)
    circle.classify()
    assert _is_surprisingly_nonadditive(circle) == solution


@pytest.mark.parametrize(
    "prop, solution",
    [
        ("num_atoms_between_r_groups", 5),
        ("ortho_classification", "Introduced"),
        *Solutions,
    ],
)
def test_update_na_dataframe(
    circle: Circle,
    prop: Props | str,
    solution: float | str,
) -> None:
    """Test nonadditivity.classification.classify:_update_na_dataframe.

    Args:
        circle (Circle): circle object
        prop (Props): property to get
        solution (float | str | bool): expected output.
    """
    nadf = pd.DataFrame()
    _update_na_dataframe(nadf, [circle])
    if isinstance(prop, str):
        assert list(nadf[prop].values) == [solution]
        return
    assert list(nadf[Circle.classification_keys[prop]].values) == [solution]


@pytest.mark.parametrize(
    "prop, solution",
    [
        (
            "num_atoms_between_r_groups",
            5,
        ),
        ("ortho_classification", "Introduced"),
        *Solutions,
    ],
)
def test_classify_circle(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
    prop: Props | str,
    solution: float | str,
) -> None:
    """Test nonadditivity.classification.classify:classify_circles.

    Args:
        transformation1 (Transformation): Transformation 1
        transformation2 (Transformation): Transformation 2
        transformation3 (Transformation): Transformation 3
        transformation4 (Transformation): Transformation 4
        prop (Props): property to get
        solution (float | str | bool): expected output.
    """
    nadf = pd.DataFrame()
    _ = classify_circles(
        [(transformation1, transformation2, transformation3, transformation4)],
        nadf,
    )
    if isinstance(prop, str):
        assert list(nadf[prop].values) == [solution]
        return
    assert list(nadf[Circle.classification_keys[prop]].values) == [solution]


@pytest.mark.parametrize(
    "prop, solution",
    [
        ("num_atoms_between_r_groups", [2, 3]),
        (Props.H_BOND_DONORS_DIFF, [0, 1]),
        (Props.H_BOND_ACCEPTORS_DIFF, [0, 1]),
        (Props.FORMAL_CHARGE_DIFF, [0, 0]),
        (Props.POLAR_SURFACE_AREA_DIFF, [0.0, 20.230000000000004]),
        (Props.NUM_ROTATABLE_BONDS_DIFF, [0, 0]),
        (Props.SP3_CARBON_DIFF, [0, 0]),
        (Props.LOG_P_DIFF, [0.8637000000000006, 1.158100000000001]),
        (Props.NUM_HEAVY_ATOMS_DIFF, [2, 3]),
        ("ortho_classification", ["None", "None"]),
        (Props.CHI0_DIFF, [1.7404869760061565, 2.610730464009235]),
        (Props.CHI1_DIFF, [0.7833624025851336, 1.1940460051080928]),
        (Props.CHI2_DIFF, [0.30279739252301363, 0.44307085693965886]),
        (Props.HAS_TERTIARY_AMIDE_FORMED, ["False", "False"]),
        (Props.HAS_INVERSION_IN_TRANSFORMATION, ["None", "None"]),
        (Props.MAX_NUM_MMPDB_CUTS, [1, 1]),
        (Props.MAX_HEAVY_ATOM_IN_TRANSFORMATION, [1, 5]),
        (Props.COMPOUND_STEREO_CLASSIFICATION, ["None", "None"]),
        (
            Props.MIN_TANIMOTO,
            (
                0.5786555786555787,
                0.5110316040548598,
                0.4996873045653534,
                0.40878552971576226,
            ),
        ),
        (Props.SUBSTITUENT_ON_SAME_RING_SYSYTEM, [False, True]),
    ],
)
def test_classify(
    per_cpd_dataframe: pd.DataFrame,
    circles: list[list[str]],
    prop: Props | str,
    solution: list[int] | list[float] | list[str] | tuple[float, float, float, float],
) -> None:
    """Test nonadditivity.classification.classify:classify.

    Args:
        per_cpd_dataframe (pd.DataFrame): per compound data
        circles (Circle): list of circles
        prop (Props): property to get
        solution (...): expected output.
    """
    per_cpd_dataframe["RDKit_Molecules"] = [
        Chem.MolFromSmiles(s)  # type: ignore pylint:disable=E1101
        for s in per_cpd_dataframe.SMILES.to_numpy()
    ]
    na_df, _, _, _ = calculate_na_output(
        circles,
        per_cpd_dataframe,
        ["TEST_PCHEMBL_VALUE"],
        include_censored=False,
    )
    classify(per_cpd_dataframe, na_df)
    if isinstance(prop, str):
        assert list(na_df[prop].values) == solution
        return
    if prop in (Props.MIN_TANIMOTO,):
        assert all(i in solution for i in na_df[Circle.classification_keys[prop]])
        return
    assert list(na_df[Circle.classification_keys[prop]].values) == solution
