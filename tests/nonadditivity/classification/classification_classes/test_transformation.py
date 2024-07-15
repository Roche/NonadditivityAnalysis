"""Test nonadditivity.classification.classifcation_classes.transformation."""

import pytest

from nonadditivity.classification import Compound, Transformation

Props = Transformation.Properties

solutions = [
    (Props.IS_H_REPLACED, True),
    (Props.NUM_HEAVY_ATOMS_IN_RGROUPS, [0, 4]),
    (Props.NUM_MMPDB_CUTS, 1),
    (Props.TERTIARY_AMIDE_FORMED, False),
    (Props.MFP2_SIMILARITY, 0.6626506024096386),
    (Props.ORTHO_SUBSTITUENT_CHANGES, 0),
    (Props.ORTHO_SUBSTITUENT_INTRODUCED, 1),
]


def test_init(
    compound1: Compound,
    compound2: Compound,
    transformation_smarts1: str,
    constant_smarts1: str,
) -> None:
    """Test Transformation:init.

    Args:
        compound1 (Compound): compound object 1
        compound2 (Compound): compound object 2
        transformation_smarts1 (str): smarts describing trasnformation
        constant_smarts1 (str): smart describing constant part.
    """
    transformation = Transformation(
        compound_1=compound1,
        compound_2=compound2,
        transformation_smarts=transformation_smarts1,
        constant_smarts=constant_smarts1,
    )
    assert transformation.compound_1_anchor == [19]
    assert list(set(transformation.compound_1_transformation_indices)) == [
        19,
        22,
    ]
    assert list(set(transformation.compound_1_constant_indices)) == list(
        range(22),
    )
    assert transformation.compound_2_anchor == [19]
    assert list(set(transformation.compound_2_transformation_indices)) == [
        19,
        22,
    ]
    assert list(set(transformation.compound_2_constant_indices)) == list(
        range(22),
    )
    del transformation


@pytest.mark.parametrize("prop, goal", solutions)
def test_get_proeprty(
    transformation2: Transformation,
    prop: Props,
    goal: bool | float | list[float],
) -> None:
    """Test Transformation:get_property.

    Args:
        transformation2 (Transformation): Transformation object
        prop (Props): property to get
        goal (bool | float | list[float]): expected value for property
    """
    assert transformation2.get_property(prop) == goal
    assert prop in transformation2._classification  # pylint: disable=W0212


@pytest.mark.parametrize("prop, goal", solutions)
def test_classify(
    transformation2: Transformation,
    prop: Props,
    goal: bool | float | list[float],
) -> None:
    """Test Transformation:classify.

    Args:
        transformation2 (Transformation): Transformation object
        prop (Props): property to get
        goal (bool | float | list[float]): expected value for property
    """
    assert not transformation2._classification  # pylint: disable=W0212
    transformation2.classify()
    assert transformation2.get_property(prop) == goal
    for member in Props:
        assert member in transformation2._classification  # pylint: disable=W0212


@pytest.mark.parametrize("prop, goal", solutions)
def test_get_classification_dict(
    transformation2: Transformation,
    prop: Props,
    goal: bool | float | list[float],
) -> None:
    """Test Transformation:get_classification_dict.

    Args:
        transformation2 (Transformation): Transformation object
        prop (Props): property to get
        goal (bool | float | list[float]): expected value for property
    """
    assert not transformation2.get_classification_dict(
        force_classification=False,
    )
    classification_dict = transformation2.get_classification_dict(
        force_classification=True,
    )
    assert classification_dict[prop] == goal
    for member in Props:
        assert member in classification_dict


@pytest.mark.parametrize("prop, goal", solutions)
def test_get_formatted_classification_dict(
    transformation2: Transformation,
    prop: Props,
    goal: bool | float | list[float],
) -> None:
    """Test Transformation:get_formatted_classification_dict.

    Args:
        transformation2 (Transformation): Transformation object
        prop (Props): property to get
        goal (bool | float | list[float]): expected value for property
    """
    assert not transformation2.get_formatted_classification_dict(
        force_classification=False,
    )
    classification_dict = transformation2.get_formatted_classification_dict(
        force_classification=True,
    )
    assert classification_dict[Transformation.classification_keys[prop]] == goal
    for member in Props:
        assert Transformation.classification_keys[member] in classification_dict


def test_get_reverse_transofrmation(transformation1: Transformation) -> None:
    """Test Transformation:get_reverse.

    Args:
        transformation1 (Transformation): transformation object.
    """
    reversed_transformation = transformation1.get_reverse()
    assert reversed_transformation.transformation_smarts == [
        transformation1.transformation_smarts[1],
        transformation1.transformation_smarts[0],
    ]
    assert reversed_transformation.constant_smarts == transformation1.constant_smarts
