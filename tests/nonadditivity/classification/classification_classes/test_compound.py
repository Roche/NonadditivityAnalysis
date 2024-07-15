"""Test nonadditivity.classification.classification_classes.compound:Compound."""

import pytest

from nonadditivity.classification import Compound
from nonadditivity.utils.types import Molecule

Props = Compound.Properties

Solutions = [
    (Props.NUM_STERO_CENTERS, 1),
    (Props.NUM_HEAVY_ATOMS, 23),
    (Props.NUM_ORTHO_CONFIGURATIONS, 4),
    (Props.HAS_UNASSIGNED_STEREOCENTERS, False),
    (
        Props.ORTHO_INDICES,
        [
            ({2, 3, 4, 5, 6, 16, 17, 18, 19}, {3, 4, 5, 6, 16, 17}),
            ({1, 2, 3, 4, 5, 6, 16, 17, 18}, {3, 4, 5, 6, 16, 17}),
            ({7, 8, 9, 10, 11, 12, 13, 14}, {8, 9, 10, 11, 12}),
            ({6, 7, 8, 9, 10, 11, 12, 13}, {8, 9, 10, 11, 12}),
        ],
    ),
    (Props.AROMATIC_INDICES, {3, 4, 5, 6, 8, 9, 10, 11, 12, 16, 17}),
    (Props.FUSED_RING_INDICES, []),
    (Props.CHARGE, 0),
    (Props.NUM_HBA, 2),
    (Props.NUM_HBD, 1),
    (Props.TPSA, 33.370000000000005),
    (Props.LOGP, 5.265200000000005),
    (Props.CHI0, 16.872032720186702),
    (Props.CHI1, 10.92516481473553),
    (Props.CHI2, 6.468033721680204),
    (Props.NUM_ROT_BONDS, 8),
    (Props.NUM_SP3_CARBONS, 10),
]


def test_init(
    compound_id1: str,
    compound_smiles1: str,
    compound_molecule1: Molecule,
) -> None:
    """Test nonadditivity.classification_classes.compound:init.

    Args:
        compound_id1 (str): test id
        compound_smiles1 (str): compound smiles
        compound_molecule1 (Molecule): compound rdkit molecule.
    """
    compound = Compound(
        molecule=compound_molecule1,
        compound_id=compound_id1,
        smiles=compound_smiles1,
    )
    assert not compound._classification  # pylint: disable=W0212
    assert compound.Properties == Compound.Properties
    for member in Props:
        assert member in compound.classification_function


@pytest.mark.parametrize("prop, goal", Solutions)
def test_get_proeprty(
    compound1: Compound,
    prop: Props,
    goal: int | bool | float | tuple[set[int], ...],
) -> None:
    """Test nonadditivity.classification_classes.Compound:get_property.

    Args:
        compound1 (Compound): compound object
        prop (Props): property to get
        goal (int | bool | float | tuple[set[int], ...]): expected value.
    """
    assert compound1.get_property(prop) == goal
    assert prop in compound1._classification  # pylint: disable=W0212


@pytest.mark.parametrize("prop, goal", Solutions)
def test_classify(
    compound1: Compound,
    prop: Props,
    goal: int | bool | float | tuple[set[int], ...],
) -> None:
    """Test nonadditivity.classification_classes.Compound:classify.

    Args:
        compound1 (Compound): compound object
        prop (Props): property to get
        goal (int | bool | float | tuple[set[int], ...]): expected value.
    """
    assert not compound1._classification  # pylint: disable=W0212
    compound1.classify()
    assert compound1._classification[prop] == goal  # pylint: disable=W0212
    for member in Props:
        assert member in compound1._classification  # pylint: disable=W0212


@pytest.mark.parametrize("prop, goal", Solutions)
def test_get_classification_dict(
    compound1: Compound,
    prop: Props,
    goal: int | bool | float | tuple[set[int], ...],
) -> None:
    """Test nonadditivity.classification_classes.Compound:get_classification_dict.

    Args:
        compound1 (Compound): compound object
        prop (Props): property to get
        goal (int | bool | float | tuple[set[int], ...]): expected value.
    """
    assert not compound1.get_classification_dict(force_classification=False)
    classification_dict = compound1.get_classification_dict(
        force_classification=True,
    )
    assert classification_dict[prop] == goal
    for member in Props:
        assert member in classification_dict


@pytest.mark.parametrize("prop, goal", Solutions)
def test_get_formatted_classification_dict(
    compound1: Compound,
    prop: Props,
    goal: int | bool | float | tuple[set[int], ...],
) -> None:
    """Test nonadditivity.classification_classes.Compound:get_formatted_classif...

    Args:
        compound1 (Compound): compound object
        prop (Props): property to get
        goal (int | bool | float | tuple[set[int], ...]): expected value.
    """
    assert not compound1.get_formatted_classification_dict(
        force_classification=False,
    )
    classification_dict = compound1.get_formatted_classification_dict(
        force_classification=True,
    )
    assert classification_dict[Compound.classification_keys[prop]] == goal
    for member in Props:
        assert Compound.classification_keys[member] in classification_dict
