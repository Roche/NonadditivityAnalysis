"""Test nonadditivity.classification.classifcation_classes.circle:Circle."""

import pytest

from nonadditivity.classification import Circle, Transformation

Props = Circle.Properties

Solutions = [
    (Props.DISTANCE_BETWEEN_R_GROUPS, 5),
    (Props.H_BOND_DONORS_DIFF, 1),
    (Props.H_BOND_ACCEPTORS_DIFF, 1),
    (Props.FORMAL_CHARGE_DIFF, 0),
    (Props.POLAR_SURFACE_AREA_DIFF, 20.230000000000004),
    (Props.NUM_ROTATABLE_BONDS_DIFF, 2),
    (Props.SP3_CARBON_DIFF, 3),
    (Props.LOG_P_DIFF, 0.3460000000000001),
    (Props.CHI0_DIFF, 3.1547005383792524),
    (Props.CHI1_DIFF, 1.8045304526403108),
    (Props.CHI2_DIFF, 1.3427031358353583),
    (Props.NUM_HEAVY_ATOMS_DIFF, 4),
    (Props.HAS_TRANSFORMATION_AT_ORTHO, 0),
    (Props.HAS_ORTHO_SUBSTITUENT_INTRODUCED, 1),
    (Props.HAS_TERTIARY_AMIDE_FORMED, "False"),
    (Props.HAS_INVERSION_IN_TRANSFORMATION, "None"),
    (Props.MAX_NUM_MMPDB_CUTS, 1),
    (Props.MAX_HEAVY_ATOM_IN_TRANSFORMATION, 4),
    (Props.COMPOUND_STEREO_CLASSIFICATION, "Unassigned"),
    (Props.SCAFFOLD, "CCCc1occc1Cc1cc(C[C@@](C)(O)[*])c(CCC)c([*])c1"),
    (Props.MIN_TANIMOTO, 0.6626506024096386),
    (Props.SUBSTITUENT_ON_SAME_RING_SYSYTEM, False),
]


def test_init(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
) -> None:
    """Test Circle:init.

    Args:
        transformation1 (Transformation): Transformation 1
        transformation2 (Transformation): Transformation 2
        transformation3 (Transformation): Transformation 3
        transformation4 (Transformation): Transformation 4
    """
    circle = Circle(
        transformation_1=transformation1,
        transformation_2=transformation2,
        transformation_3=transformation3,
        transformation_4=transformation4,
    )
    assert not circle._classification  # pylint:disable=W0212
    del circle


def test_get_compounds(
    circle: Circle,
) -> None:
    """Test Circle:get_compounds.

    Args:
        circle (Circle): circle to get compounds from
    """
    assert circle.compounds() == [
        circle.transformation_1.compound_1,
        circle.transformation_1.compound_2,
        circle.transformation_4.compound_2,
        circle.transformation_4.compound_1,
    ]


@pytest.mark.parametrize("prop, goal", Solutions)
def test_get_proeprty(circle: Circle, prop: Props, goal: float | str) -> None:
    """Test Circle:get_property.

    Args:
        circle (Circle): circle object
        prop (Props): property to get
        goal (float | str): test value
    """
    assert circle.get_property(prop) == goal
    assert prop in circle._classification  # pylint: disable=W0212


@pytest.mark.parametrize("prop, goal", Solutions)
def test_classify(circle: Circle, prop: Props, goal: float | str) -> None:
    """Test Circle:classify.

    Args:
        circle (Circle): circle object
        prop (Props): property to get
        goal (float | str): test value
    """
    assert not circle._classification  # pylint: disable=W0212
    circle.classify()
    assert circle._classification[prop] == goal  # pylint: disable=W0212
    for member in Props:
        assert member in circle._classification  # pylint: disable=W0212


@pytest.mark.parametrize("prop, goal", Solutions)
def test_get_classification_dict(
    circle: Circle,
    prop: Props,
    goal: float | str,
) -> None:
    """Test Circle:get_classification_dict.

    Args:
        circle (Circle): circle object
        prop (Props): property to get
        goal (float | str): test value
    """
    assert not circle.get_classification_dict(force_classification=False)
    classification_dict = circle.get_classification_dict(
        force_classification=True,
    )
    assert classification_dict[prop] == goal
    for member in Props:
        assert member in classification_dict


@pytest.mark.parametrize("prop, goal", Solutions)
def test_get_formatted_classification_dict(
    circle: Circle,
    prop: Props,
    goal: float | str,
) -> None:
    """Test Circle:get_formatted_classification_dict.

    Args:
        circle (Circle): circle object
        prop (Props): property to get
        goal (float | str): test value
    """
    assert not circle.get_formatted_classification_dict(
        force_classification=False,
    )
    classification_dict = circle.get_formatted_classification_dict(
        force_classification=True,
    )
    assert classification_dict[Circle.classification_keys[prop]] == goal
    for member in Props:
        assert Circle.classification_keys[member] in classification_dict
