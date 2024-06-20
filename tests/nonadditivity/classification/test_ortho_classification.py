"""Test nonadditivity.classification.ortho_classification."""
import pytest
from rdkit import Chem

from nonadditivity.classification import Circle, Compound
from nonadditivity.classification.ortho_classification import (
    _atoms_are_neighbors,
    _is_transformation_at_ortho,
    _ortho_is_bicycle,
    differentiate_r_groups,
    get_correct_ring_match,
    get_num_ortho_configurations,
    get_ortho_indices,
    get_ortho_transformation_changing,
    get_ortho_transformation_introduced,
    match_ortho_patterns,
    ortho_substituent_exchanged,
    ortho_substituent_introduced,
    substituent_indices_in_same_ring,
    transformation_at_ortho,
)
from nonadditivity.utils.types import Molecule
from tests._utils import raises_exceptions


def test_get_correct_ring_match() -> None:
    """Test nonadditivity.classification.ortho_classification:get_correct_ring_match."""
    wholematch = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    wrong_ring_place = (0, 1, 2, 3, 4, 5)
    wrong_ring_size = (0, 1, 2)
    correct_ring = (3, 4, 5, 6, 7, 8)
    assert get_correct_ring_match(
        whole_match=wholematch,
        ring_matches=(wrong_ring_place, wrong_ring_size, correct_ring),
    ) == set(correct_ring)
    assert get_correct_ring_match(whole_match=wholematch, ring_matches=()) is None


def test_match_ortho_patterns(compound_molecule1: Molecule) -> None:
    """Test nonadditivity.classification.ortho_classification:match_ortho_patterns.

    Args:
        compound_molecule1 (Molecule): test molecule
    """
    with pytest.raises(AttributeError):
        match_ortho_patterns(molecule="wrong type")
    assert match_ortho_patterns(compound_molecule1) == [
        ({2, 3, 4, 5, 6, 16, 17, 18, 19}, {3, 4, 5, 6, 16, 17}),
        ({1, 2, 3, 4, 5, 6, 16, 17, 18}, {3, 4, 5, 6, 16, 17}),
        ({7, 8, 9, 10, 11, 12, 13, 14}, {8, 9, 10, 11, 12}),
        ({6, 7, 8, 9, 10, 11, 12, 13}, {8, 9, 10, 11, 12}),
    ]
    assert match_ortho_patterns(Chem.MolFromSmiles("c1c(C)cc(CC)cc1")) == []


def test_get_num_ortho_configurations(compound_molecule1: Molecule) -> None:
    """Test ...:get_num_ortho_configurations.

    Args:
        compound_molecule1 (Molecule): test molecule
    """
    raises_exceptions(
        test_object=compound_molecule1,
        function=get_num_ortho_configurations,
        exception=AttributeError,
        molecule="wrongtype",
    )
    assert get_num_ortho_configurations(molecule=compound_molecule1) == 4
    assert (  # pylint: disable=E1101
        get_num_ortho_configurations(molecule=Chem.MolFromSmiles("c1c(C)cc(CC)cc1"))
        == 0
    )
    assert (
        get_num_ortho_configurations(  # pylint: disable=E1101
            molecule=Chem.MolFromSmiles("C1C(C)CC(CC)CC1"),
        )
        == 0
    )


def test_get_ortho_indices(compound_molecule1: Molecule) -> None:
    """Test nonadditivity.classification.ortho_classification:get_ortho_indices.

    Args:
        compound_molecule1 (Molecule): test molecule
    """
    raises_exceptions(
        test_object=compound_molecule1,
        function=get_ortho_indices,
        exception=AttributeError,
        molecule="wrongtype",
    )
    assert get_ortho_indices(molecule=compound_molecule1) == [
        ({2, 3, 4, 5, 6, 16, 17, 18, 19}, {3, 4, 5, 6, 16, 17}),
        ({1, 2, 3, 4, 5, 6, 16, 17, 18}, {3, 4, 5, 6, 16, 17}),
        ({7, 8, 9, 10, 11, 12, 13, 14}, {8, 9, 10, 11, 12}),
        ({6, 7, 8, 9, 10, 11, 12, 13}, {8, 9, 10, 11, 12}),
    ]
    assert get_ortho_indices(molecule=Chem.MolFromSmiles("c1c(C)cc(CC)cc1")) == []


def test_differentiate_r_groups(compound1: Compound) -> None:
    """Test nonadditivity.classification.ortho_classification:differentiate_r_groups.

    Args:
        compound1 (Compound): test compound
    """
    same, other = differentiate_r_groups(
        compound=compound1,
        non_ring_indices={2, 18, 19},
    )
    assert other == 2
    assert same == {18, 19}
    with pytest.raises(AssertionError):
        _ = differentiate_r_groups(
            compound=compound1,
            non_ring_indices={2, 3},
        )


@pytest.mark.parametrize(
    "ind, sol",
    [({5, 16, 17}, True), ({8, 16, 17}, False)],
)
def test_substituent_indices_in_same_ring(
    compound1: Compound,
    ind: set[int],
    sol: bool,
) -> None:
    """Test ...:substituent_indices_in_same_ring.

    Args:
        compound1 (Compound): test compound
        ind (set[int]): test indices
        sol (bool): expected output
    """
    assert (
        substituent_indices_in_same_ring(
            compound=compound1,
            non_ring_indices=ind,
        )
        == sol
    )

    with pytest.raises(AssertionError):
        _ = substituent_indices_in_same_ring(
            compound=compound1,
            non_ring_indices={2, 3},
        )


def test_ortho_is_bicycle(compound1: Compound) -> None:
    """Test nonadditivity.classification.ortho_classification.

    Args:
        compound1 (Compound): test compound
    """
    assert not _ortho_is_bicycle(
        compound=compound1,
        non_ring_indices={2, 18, 19},
        ring_indices={3, 4, 5, 6, 16, 17},
    )
    assert _ortho_is_bicycle(
        compound=Compound(
            smiles="c1c2ccccc2c(F)cc1",
            molecule=Chem.MolFromSmiles("c1c2ccccc2c(F)cc1"),  # type: ignore pylint: disable=E1101
            compound_id="1",
        ),
        non_ring_indices={5, 4, 8},
        ring_indices={0, 1, 6, 7, 9, 10},
    )
    with pytest.raises(AssertionError):
        _ = _ortho_is_bicycle(
            compound=compound1,
            non_ring_indices={2, 9},
            ring_indices={3, 4, 5, 6, 16, 17},
        )


@pytest.mark.parametrize(
    "ind, sol",
    [({0, 1}, True), ({3, 4}, True), ({17, 18}, True), ({17, 19}, False)],
)
def test_atoms_are_neighbors(compound1: Compound, ind: set[int], sol: bool) -> None:
    """Test nonadditivity.classification.ortho_classification.

    Args:
        compound1 (Compound): test compound
        ind (set[int]): test indices
        sol (bool): expected solution
    """
    assert _atoms_are_neighbors(compound=compound1, indices=ind) == sol
    with pytest.raises(AssertionError):
        _atoms_are_neighbors(compound=compound1, indices={1, 2, 3})


@pytest.mark.parametrize(
    "ci, sol",
    [
        (list(range(18)), 1),
        ([6, 7, 8, 9, 10, 11, 12], 1),
        ([3, 4, 5, 6, 16, 17], 0),
        ([], -1),
    ],
)
def test_is_transformation_at_ortho(
    compound1: Compound,
    ci: list[int],
    sol: int,
) -> None:
    """Test nonadditivity.classification.ortho_classification.

    Args:
        compound1 (Compound): test compound
        ci (list[int]): test indices
        sol (int): expected solution
    """
    assert (
        _is_transformation_at_ortho(
            compound=compound1,
            constant_indices=ci,
        )
        == sol
    )


@pytest.mark.parametrize(
    "compounds, ts, ind, solutions",
    [
        (
            ("compound1", "compound2"),
            "transformation_smarts1",
            [
                list(range(22)),
                list(range(22)),
            ],
            (0, 0, 0),
        ),
        (
            ("compound1", "compound4"),
            "transformation_smarts2",
            [
                list(set(range(23))),
                list(set(range(27)) - {12, 13, 14, 15}),
            ],
            (1, 1, 0),
        ),
        (
            ("compound1", "compound4"),
            "transformation_smarts2",
            [
                [],
                [[]],
            ],
            (-1, -1, -1),
        ),
    ],
)
def test_transformation_at_ortho(
    compounds: tuple[str, str],
    ts: str,
    ind: list[list[int]],
    solutions: tuple[int, int, int],
    request: pytest.FixtureRequest,
) -> None:
    """Test nonadditivity.classification.ortho_classification.

    Args:
        compounds: (tuple[str, str]): test compound fixture names
        ts (str): transformation smarts
        ind (list[list[int]]): indices
        solutions: (tuple[int, int, int]): expected values
        request (pytest.FixtureRequest): pytest magic
    """
    with pytest.raises(KeyError):
        transformation_at_ortho(wrong_keyword=1)
    compounds = [request.getfixturevalue(c) for c in (compounds)]
    assert (
        transformation_at_ortho(
            compounds=compounds,
            constant_indices=ind,
        )
        == solutions[0]
    )
    assert (
        ortho_substituent_introduced(
            compounds=compounds,
            constant_indices=ind,
            transformation_smarts=request.getfixturevalue(ts).split(">>"),
        )
    ) == solutions[1]
    assert (
        ortho_substituent_exchanged(
            compounds=compounds,
            constant_indices=ind,
            transformation_smarts=request.getfixturevalue(ts).split(">>"),
        )
    ) == solutions[2]


def test_get_ortho_transformation_changing(circle: Circle) -> None:
    """Test nonadditivity.classification.ortho_classification.

    Args:
        circle (Circle): test circle
    """
    assert not get_ortho_transformation_changing(
        transformation_1=circle.transformation_1,
        transformation_2=circle.transformation_2,
        transformation_3=circle.transformation_3,
        transformation_4=circle.transformation_4,
    )


def test_get_ortho_transformation_introduced(circle: Circle) -> None:
    """Test nonadditivity.classification.ortho_classification.

    Args:
        circle (Circle): test circle
    """
    assert get_ortho_transformation_introduced(
        transformation_1=circle.transformation_1,
        transformation_2=circle.transformation_2,
        transformation_3=circle.transformation_3,
        transformation_4=circle.transformation_4,
    )
