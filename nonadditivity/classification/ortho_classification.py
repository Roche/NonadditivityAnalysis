"""Implement Ortho classification.

Conains the code that manages the classification of ortho effects over
all the compound, transformation and circle level of the classification
workflow of the Nonadditivity Analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem

from nonadditivity.classification.transformation_classification import is_h_replaced
from nonadditivity.classification.utils import list_empty

if TYPE_CHECKING:
    from nonadditivity.classification import Compound, Transformation
    from nonadditivity.utils.types import Molecule


# ===========================================================
#
#           COMPOUND ORTHO FUNCTIONS
#
# ===========================================================


def get_correct_ring_match(
    whole_match: tuple[int],
    ring_matches: tuple[tuple[int], ...],
) -> set[int] | None:
    """Return the ring match that is included in whole match.

    Args:
        whole_match (set[int]): set to match ring to
        ring_matches (tuple[tuple[int]]): list of ring matches

    Returns:
        set[int] | None: correct ring match
    """
    whole_set = set(whole_match)
    ring_size = len(whole_set) - 3
    for ring_match in ring_matches:
        ring_match_set = set(ring_match)
        if len(whole_set.intersection(ring_match_set)) == ring_size:
            return ring_match_set
    return None


def match_ortho_patterns(
    molecule: Molecule,
) -> list[tuple[set[int], ...]]:
    """Find Ortho substituted aromatic rings in a molecule.

    Takes a molecule and matches the following to smarts:
    '[A,a]a1a([A,a][A,a])aaaa1', '[A,a]a1a([A,a][A,a])aaa1'.
    It returns tuples containing the matches as well as the
    matches for only the ring part of the patterns.

    Args:
        molecule (Molecule): molecule to match ortho patterns

    Returns:
        list[tuple[set[int], Optional[set[int]]]]: list of tuples
        containing the ortho match as well as only the ring part
        that is part of the ortho match.
    """
    ortho_patterns = [
        (
            Chem.MolFromSmarts(patt[0]),  # type: ignore pylint: disable=E1101
            Chem.MolFromSmarts(patt[1]),  # type: ignore pylint: disable=E1101
        )
        for patt in [
            ("[A,a]a1a([A,a][A,a])aaaa1", "a1aaaaa1"),
            ("[A,a]a1a([A,a][A,a])aaa1", "a1aaaa1"),
        ]
    ]
    matches = []
    for pattern, ring_pattern in ortho_patterns:
        whole_matches = molecule.GetSubstructMatches(pattern)
        ring_matches = molecule.GetSubstructMatches(ring_pattern)
        for match in whole_matches:
            ring_match = get_correct_ring_match(
                whole_match=match,
                ring_matches=ring_matches,
            )
            if ring_match is None:
                continue
            matches.append((set(match), ring_match))
    return matches


def get_num_ortho_configurations(**kwargs: Molecule) -> int:
    """Get number of ortho matches.

    Returns number of aromatic rings with ortho substituents
    in a molecule. Can return up to 2n of n ortho configurations
    since some of them are matched twice.

    use keyword argument 'molecule' for it to work.

    Returns:
        int: Num of ortho substituents in molecule.
    """
    return len(get_ortho_indices(**kwargs))


def get_ortho_indices(**kwargs: Molecule) -> list[tuple[set[int], ...]]:
    """Get indices of possible ortho substituents.

    Gets the indices of atoms in a molecule that are
    part of ortho substituted aromatic rings, including
    the rings and 1, respectively 2 atoms of the substituents.
    together with the whole match it returns the indices of just
    the ring that is matched in each ortho match.

    use keyword argument 'molecule' for it to work.

    Returns:
        list[tuple[set[int],set[int]]]: tuple of (ortho_match, ring_match)
    """
    molecule: Molecule = kwargs["molecule"]
    return match_ortho_patterns(molecule)


# ===========================================================
#
#           TRANSFORMATION ORTHO FUNCTIONS
#
# ===========================================================


def differentiate_r_groups(
    non_ring_indices: set[int],
    compound: Compound,
) -> tuple[set[int], int | None]:
    """Differentiat indices of bigger and smaller r group.

    Returns tuple of a set of the indices that are in the bigger r group
    and the single index of the atom in the smaller r group.

    Args:
        non_ring_indices (set[int]): set of indices not in the aromatic
        ring investigated.
        compound (Compound): compound under investigation

    Returns:
        tuple[set[int], int | None]: set[indices longer r grouop],
        index shorter r group
    """
    assert len(non_ring_indices) == 3
    same_side = set()
    lonely_side = None
    for non_ring_index in non_ring_indices:
        neighbors = {
            atom.GetIdx()
            for atom in compound.rdkit_molecule.GetAtomWithIdx(
                non_ring_index,
            ).GetNeighbors()
        }
        if neighbors.intersection(non_ring_indices):
            same_side.add(non_ring_index)
            continue
        lonely_side = non_ring_index
    return same_side, lonely_side


def substituent_indices_in_same_ring(
    non_ring_indices: set[int],
    compound: Compound,
) -> bool:
    """Check whether both substituents are in same ring.

    Checks wheter substituents R1 and R2 in a pattern like R1-*-*-R2 are
    in the same ring.

    Args:
        non_ring_indices (set[int]): indices of R1 and R2
        compound (Compound): the compound in which this occurs.

    Returns:
        bool: True if R1 and R2 are in the same ring.
    """
    same_side, lonely_side = differentiate_r_groups(
        non_ring_indices=non_ring_indices,
        compound=compound,
    )
    ring_info = compound.rdkit_molecule.GetRingInfo()
    sameside = False
    for same in same_side:
        if ring_info.AreAtomsInSameRing(same, lonely_side):
            sameside = True
    return sameside


def _ortho_is_bicycle(
    non_ring_indices: set[int],
    ring_indices: set[int],
    compound: Compound,
) -> bool:
    """Check whether ortho pattern is a bicycle.

    Returns true if the longer substituent is part
    of a bicycle with the aromatic ring.

    Args:
        non_ring_indices (set[int]): indices not in ring
        ring_indices (set[int]): ring indices
        compound (Compound): compound object

    Returns:
        bool: true if bicycle
    """
    same_side, _ = differentiate_r_groups(
        non_ring_indices=non_ring_indices,
        compound=compound,
    )
    neighbors = [
        atom.GetIdx()
        for index in same_side
        for atom in compound.rdkit_molecule.GetAtomWithIdx(
            index,
        ).GetNeighbors()
    ]
    anchor = ring_indices.intersection(set(neighbors)).pop()
    assert isinstance(anchor, int)
    ring_info = compound.rdkit_molecule.GetRingInfo()
    return any(ring_info.AreAtomsInSameRing(s, anchor) for s in same_side)


def _atoms_are_neighbors(compound: Compound, indices: set[int]) -> bool:
    """Check whether atoms are neighbours.

    Returns if two atoms with indices given in 'indices' are neighbors in a
    compound.

    Args:
        compound (Compound): Compound
        indices (set[int]): 2 atom indices

    Returns:
        bool: True if atoms are connected via 1 bond.
    """
    assert len(indices) == 2
    return indices.pop() in [
        atom.GetIdx()
        for atom in compound.rdkit_molecule.GetAtomWithIdx(
            indices.pop(),
        ).GetNeighbors()
    ]


def _is_transformation_at_ortho(
    compound: Compound,
    constant_indices: list[int] | list[list[int]],
) -> int:
    """Check whether transformation is at ortho.

    Checks wheter a transformation is happening at the ortho position
    in an aromatic ring.

    Args:
        compound (Compound): compound to get ortho indices from
        constant_indices (list[int] | list[list[int]]): constant ondices of compound

    Returns:
        int: 1 == yes
             0 == no
              -1 == not identifiable (due to ambiguosly defined constant_indices)
    """
    ortho_match_found = False
    if list_empty(constant_indices) or isinstance(constant_indices[0], list):
        return -1
    ortho_matches = compound.get_property(compound.Properties.ORTHO_INDICES)
    assert isinstance(ortho_matches, list)
    constant_set = set(constant_indices)
    for ortho_match, ring_match in ortho_matches:
        ring_size = len(ring_match)
        # Check that ring is in constant part
        if len(constant_set.intersection(ring_match)) != ring_size:
            continue
        non_ring_indices = ortho_match.symmetric_difference(ring_match)
        const_substituents_intersect = constant_set.intersection(
            non_ring_indices,
        )
        # Check that longer substituent is also part of constant part
        if len(const_substituents_intersect) != 2:
            continue
        # Check that not one of either r group is matched with constant part.
        if not _atoms_are_neighbors(
            compound=compound,
            indices=const_substituents_intersect,  # type:ignore
        ):
            continue
        # Check that the two r groups are not in the same ring
        if substituent_indices_in_same_ring(
            non_ring_indices=non_ring_indices,
            compound=compound,
        ):
            continue
        # Check that the longer substituent is not in a bisycle with the ring match.
        if _ortho_is_bicycle(
            non_ring_indices=non_ring_indices,
            ring_indices=ring_match,
            compound=compound,
        ):
            continue
        ortho_match_found = True
    return int(ortho_match_found)


def transformation_at_ortho(**kwargs) -> int:
    """Check whether transformation happens at ortho.

    Looks at both sides of a transformation and tries to identify
    whether the transformation happens at the ortho position of an
    aromatic ring.

    provide keyword arguments 'compounds' and 'constant_indices'
    for this function to work.

    Returns:
        int: 1 == yes
             0 == no
              -1 == not identifiable (due to ambiguosly defined constant_indices)
    """
    ret = 0
    for compound, constant in zip(
        kwargs["compounds"],
        kwargs["constant_indices"],
    ):
        is_at_ortho = _is_transformation_at_ortho(
            compound=compound,
            constant_indices=constant,
        )
        if is_at_ortho == 1:
            return 1
        if is_at_ortho == -1:
            ret = -1
    return ret


def ortho_substituent_introduced(**kwargs) -> int:
    """Check whether ortho substituent is introduced.

    Determines wheter a substituent is introdced/removed at an ortho
    position of an aromatic ring in the molecule.

    provide keyword arguments 'compounds' and 'constant_indices' for
    this function to work

    Returns:
        int: 1 == yes
             0 == no
              -1 == not identifiable (due to ambiguosly defined constant_indices)
    """
    ortho = transformation_at_ortho(**kwargs)
    if ortho < 1:
        return ortho
    return int(is_h_replaced(**kwargs))


def ortho_substituent_exchanged(**kwargs) -> int:
    r"""Check whether ortho substituent is exchanged.

    Determines wheter a substituent is exchanged at an ortho position
    of an aromatic ring in the molecule

    provide keyword arguments 'compounds' and 'constant_indices' for
    this function to work.

    Returns:
        int: 1 == yes
            0 == no
             -1 == not identifiable (due to ambiguosly defined constant_indices)
    """
    ortho = transformation_at_ortho(**kwargs)
    if ortho < 1:
        return ortho
    return not int(is_h_replaced(**kwargs))


# ===========================================================
#
#           CIRCLE ORTHO FUNCTIONS
#
# ===========================================================


def get_ortho_transformation_changing(**kwargs: Transformation) -> bool:
    """Check whether circle has transformation at ortho position.

    Checks all transformation in a circle on wheter any of the transformations
    happens at the ortho position of an aromatic ring.

    Returns:
        bool: True if transformation at ortho position.
    """
    for i in range(4):
        transformation = kwargs[f"transformation_{i+1}"]
        if (
            transformation.get_property(
                transformation.Properties.ORTHO_SUBSTITUENT_CHANGES,
            )
            == 1
        ):
            return True
    return False


def get_ortho_transformation_introduced(**kwargs: Transformation) -> bool:
    """Check whether circle has introduction of ortho substituent.

    Checks all transformation in a circle on wheter any of the transformations
    happens at the ortho position of an aromatic ring.

    Returns:
        bool: True if transformation at ortho position.
    """
    for i in range(4):
        transformation = kwargs[f"transformation_{i+1}"]
        if (
            transformation.get_property(
                transformation.Properties.ORTHO_SUBSTITUENT_INTRODUCED,
            )
            == 1
        ):
            return True
    return False
