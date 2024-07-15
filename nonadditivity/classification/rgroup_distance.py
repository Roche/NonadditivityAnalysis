"""Functionality for r group distance calculation.

Contains all the functionality for evalutating the distance between two
r groups of a double transformation cycle over all levels (compound,
transformation and circle). This is part of the classification workflow
of the Nonadditivity Analysis.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import rdmolfiles

from nonadditivity.classification.circle_classification import get_combinations
from nonadditivity.classification.utils import is_match_unique
from nonadditivity.utils.log import get_logger

if TYPE_CHECKING:
    from nonadditivity.classification import Transformation
    from nonadditivity.utils.types import Molecule

logger = get_logger()


# -------------------------------------------------------------------------
#
#               TRANSFORMATION LEVEL OF THINGS
#
# =========================================================================


def all_matches_symetrically_interchangeable(
    matches: tuple[tuple[int, ...], ...],
    molecule: Molecule,
) -> bool:
    """Check whether multiple matches are symmetyrically interchangable.

    Checks if two substructure matches are symetrically interchangeable.
    E.g., a methyl has two ortho substitued fluorines, which are both matched
    once if you want to match [c1ccccc1F]. Both matches are equivalent.

    Args:
        matches (tuple[tuple[int]]): list of possible matches
        molecule (Molecule): molecule the matches are from.

    Returns:
        bool: True if matches are symetrically interchangeable.
    """
    if len(matches) != 2:
        return False
    difference = sorted(set(matches[0]).symmetric_difference(matches[1]))
    if len(difference) % 2 != 0:
        return False
    ranks = list(
        rdmolfiles.CanonicalRankAtoms(  # pylint: disable=I1101
            molecule,
            breakTies=False,
        ),
    )
    ranks = sorted([ranks[d] for d in difference])
    equal = True
    for index in range(0, len(ranks), 2):
        if ranks[index] == ranks[index + 1]:
            continue
        equal = False
    return equal


def all_matches_chemically_equivalent(
    matches: tuple[tuple[int, ...], ...],
    molecule: Molecule,
) -> bool:
    """Check whether all matches are chemically equivalent.

    Checks wheter two matches of a substructure are chemically
    equivalent, so lets say we try to match C-F and our molecule
    has a CF3 group. This group is matched three times, even tho
    every match is equivalent for our purposes.

    Args:
        matches (tuple[tuple[int]]): Matches
        molecule (Molecule): molecule matches are found

    Returns:
        bool: True if all matches are chemically equivalent.
    """
    differences = set()
    if len(matches) == 2:
        for match in matches:
            differences = differences.symmetric_difference(set(match))
    elif len(matches) == 3:
        diff1 = set(matches[0]).symmetric_difference(set(matches[1]))
        diff2 = set(matches[0]).symmetric_difference(set(matches[2]))
        differences = diff1.union(diff2)
    else:
        return False
    if len(differences) != len(matches):
        return False
    neighbors = set()
    for index in differences:
        for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors():
            neighbors.add(neighbor.GetIdx())
    return len(neighbors) == 1


def get_correct_match(
    single_match: tuple[int, ...],
    multiple_matches: tuple[tuple[int, ...], ...],
    num_attachment_points: int,
) -> list[int] | tuple[int, ...]:
    """Get the correct match.

    You have a list A and a list of lists B and a given length of the
    intersection between the single list and one of the lists in the list of lists
    should have. returns the list from the list of lists that has the right length
    intersection with the list A.
    if no list in B has the right length intersection with A, an empty list is returned

    Args:
        single_match (list[int]): list A
        multiple_matches (tuple[tuple[int]]): list of lists B
        num_attachment_points (int): length of intersection

    Returns:
        list[int]: [] or list out of B with correct length Intersection.
    """
    single_set = set(single_match)
    for match in multiple_matches:
        if len(single_set.intersection(set(match))) == num_attachment_points:
            return match
    return []


def remove_stereo_at_beginning(smarts: str) -> str:
    """Remove stereo information from smarts.

    Removes E/Z information from cut smarts, so that
    /C/CCC becomes CCCC.

    Args:
        smarts (str): smarts to check.

    Returns:
        str: smarts without invalid stereo information
    """
    exchanged = False
    for start in ("\\", "/"):
        if smarts.startswith(start):
            smarts = smarts.lstrip(start)
            exchanged = True
            break
    if not exchanged:
        return smarts
    index = -1
    for ind, char in enumerate(smarts):
        if char in ("/", "\\"):
            index = ind
            break
    if index == -1:
        return smarts
    return smarts[:index] + smarts[index + 1 :]


def get_matches_without_hydrogen(
    constant_smarts: str,
    transformation_smarts: str,
    molecule: Molecule,
) -> tuple[list[int], ...] | tuple[list[list[int]], ...]:
    """Get matches without hydrogen.

    This function tries to uniquely match transformation_smarts and constant_smarts in
    a molecule, where no hydrogen is part of the transformation. See comments below on
    how this is achieved.

    Args:
        constant_smarts (str): constant smarts
        transformation_smarts (str): transformation smarts'
        molecule (Molecule): molecule to match smarts

    Raises:
        ValueError: Raised if a case occurrs that is not covered.

    Returns:
        tuple[list[int], ...] | tuple[list[list[int]], ...]: multiple or unique matches
        for constant and transformation smarts on  molecules.
    """
    # determine number of attachment points, needed to later identify
    # the correct transformation match, if more than one transformation
    # match is found.
    original_constant_smarts = constant_smarts
    num_attachment_points = 1
    for number in range(1, 4):
        if f"[*:{number}]" in constant_smarts:
            num_attachment_points = number
            constant_smarts = constant_smarts.replace(f"[*:{number}]", "")
            continue
        break
    constant_smarts = remove_stereo_at_beginning(smarts=constant_smarts)
    with contextlib.suppress(Exception):
        constant_matches: tuple[tuple[int]] = molecule.GetSubstructMatches(
            Chem.MolFromSmarts(  # type:ignore pylint: disable=E1101
                constant_smarts,
            ),
        )
    transformation_matches: tuple[tuple[int]] = molecule.GetSubstructMatches(
        Chem.MolFromSmarts(  # type:ignore pylint: disable=E1101
            transformation_smarts,
        ),
    )
    # start distinguishing different cases and try to get the correct match
    # for both the constant part as well as the transformation part.

    # this is a pretty bad case, since rdkit is not able to match a smarts
    # pattern that was created by an rdkit submodule. This will hopefully
    # be fixed on the rdkit side of thing in the near future.
    # an empty list for the constant match is returned, so things will be
    # taken care of on the circle level of things.

    if not constant_matches:
        return [[]], [list(match) for match in transformation_matches]

    # best possible case, both constant and transformation part are only
    # matched once. return these values

    if len(constant_matches) == len(transformation_matches) == 1:
        return list(constant_matches[0]), list(transformation_matches[0])

    # second best cases follow, by having either just one transformation
    # or constant match, the correct partner can be found by looking
    # for the pair that has an len(intersection of indices)==numattachmentpoints.

    if len(constant_matches) == 1 and len(transformation_matches) > 1:
        constant_match = list(constant_matches[0])
        transformation_match = get_correct_match(
            single_match=constant_match,
            multiple_matches=transformation_matches,
            num_attachment_points=num_attachment_points,
        )
        return constant_match, transformation_match

    if len(constant_matches) > 1 and len(transformation_matches) == 1:
        transformation_match = list(transformation_matches[0])
        constant_match = get_correct_match(
            single_match=transformation_match,
            multiple_matches=constant_matches,
            num_attachment_points=num_attachment_points,
        )
        return constant_match, transformation_match

    # now things get tricky. If both the constant as well as the
    # transformation pattern are matched more than once, there
    # are different cases that need to be distinguished, and
    # some that cannot be solved on this level of the classification
    # and will need reevaluation on the circle level of things.

    if len(constant_matches) > 1 and len(transformation_matches) > 1:
        # first case is going to be easy, this would be when
        # more than one chemically equivalent atom is attached
        # to the same atom and hence the constant part is
        # matched multiple times. In this case, you can take
        # any match since the important index is the one,
        # the r group is attached to which in every match,
        # should be the same.

        if all_matches_chemically_equivalent(
            matches=constant_matches,
            molecule=molecule,
        ):
            constant_match = list(constant_matches[0])
            transformation_match = get_correct_match(
                single_match=constant_match,
                multiple_matches=transformation_matches,
                num_attachment_points=num_attachment_points,
            )
            return constant_match, transformation_match

        # this second case is more worrysome, since here we are talking
        # cases that cannot be handled on the transformation level. here
        # we are talking cases were atoms are symetrically equivalent but
        # not bound to the same atom.

        if all_matches_symetrically_interchangeable(
            matches=constant_matches,
            molecule=molecule,
        ):
            return [list(m) for m in constant_matches], [
                list(m) for m in transformation_matches
            ]

    newmatch = molecule.GetSubstructMatches(
        Chem.MolFromSmarts(  # type:ignore pylint: disable=E1101
            original_constant_smarts,
        ),
    )
    if len(newmatch) == 1:
        matchset = set(newmatch[0])
        bettermatch = []
        for match in constant_matches:
            newset = set(match)
            if (
                len(newset.intersection(matchset))
                == len(newset) + num_attachment_points
            ):
                bettermatch = list(match)
                break
        transformation_match = get_correct_match(
            single_match=bettermatch,
            multiple_matches=transformation_matches,
            num_attachment_points=num_attachment_points,
        )
        return bettermatch, transformation_match
    return [list(m) for m in constant_matches], [
        list(m) for m in transformation_matches
    ]


def get_matches_with_hydrogen(
    constant_smarts: str,
    transformation_smarts: str,
    molecule: Molecule,
) -> tuple[list[int], ...] | tuple[list[list[int]], ...]:
    """Get match with hydrogen.

    This function tries to uniquely match transformation_smarts and constant_smarts in
    a molecule, where a hydrogen is part of the transformation. See comments below on
    how this is achieved.

    Args:
        constant_smarts (str): constant smarts
        transformation_smarts (str): transformation smarts'
        molecule (Molecule): molecule to match smarts

    Raises:
        ValueError: Raised if a case occurrs that is not covered.

    Returns:
        tuple[list[int], ...] | tuple[list[list[int]], ...]: multiple or unique matches
        for constant and transformation smarts on molecules.
    """

    def remove_stereochem(smarts: str) -> str:
        return smarts.replace("[*:1][C@H]", "[*:1]C").replace("[*:1][C@@H]", "[*:1]C")

    # same shenanigans but with hydrogens. since the transformation part
    # that is the hydrogens is matched all around the molecule, we leave
    # the [*:1] on the constant part first, so we can find the transformation
    # by looking for the match that has an len(intersection) == 2.
    mol_w_h = Chem.AddHs(molecule)  # type: ignore pylint: disable=E1101
    constant_smarts = remove_stereochem(constant_smarts)
    transformation_matches = mol_w_h.GetSubstructMatches(
        Chem.MolFromSmarts(transformation_smarts),  # type: ignore pylint: disable=E1101
    )
    constant_matches_w_h = mol_w_h.GetSubstructMatches(
        Chem.MolFromSmarts(constant_smarts),  # type: ignore pylint: disable=E1101
    )

    # what is actually returned in the end, since for the constant part
    # we only want the constant part later on.
    const_without_anchor = Chem.MolFromSmarts(  # type: ignore pylint: disable=E1101
        remove_stereo_at_beginning(constant_smarts.replace("[*:1]", "")),
    )
    if const_without_anchor is None or not const_without_anchor:
        return [[]], [list(m) for m in transformation_matches]
    constant_matches = molecule.GetSubstructMatches(
        const_without_anchor,
    )
    # Not able to solve this problem on this level of things,
    # since rdkit is not able to distinguish between chiral
    # hidrogens if two of them are attached to a carbon. This
    # is solved by looking at another compound on the circle level.

    if not constant_matches_w_h:
        return [[]], [list(m) for m in transformation_matches]

    # optimal case. find the correct hydrogen by looking for the match
    # that has a len(intersection) of 2

    if len(constant_matches_w_h) == 1:
        constant_match = list(constant_matches[0])
        transformation_match = get_correct_match(
            single_match=list(constant_matches_w_h[0]),
            multiple_matches=transformation_matches,
            num_attachment_points=2,
        )
        return constant_match, transformation_match

    # again gets tricky like it was the case without the hydrogens

    # if the different matches are all around the same CH3 for
    # example, we are only interested in the C.

    if (
        all_matches_chemically_equivalent(
            matches=constant_matches_w_h,
            molecule=mol_w_h,
        )
        and len(constant_matches) == 1
    ):
        constant_match = list(constant_matches[0])
        transformation_match = get_correct_match(
            single_match=list(constant_matches_w_h[0]),
            multiple_matches=transformation_matches,
            num_attachment_points=2,
        )
        return constant_match, transformation_match

    # again the case, where we have symetrically equivalent atoms,
    # we need to solve the problem on the next higher level, which
    # is going to be the circle level.

    if all_matches_symetrically_interchangeable(
        matches=constant_matches_w_h,
        molecule=mol_w_h,
    ):
        return [list(m) for m in constant_matches_w_h], [
            list(m) for m in transformation_matches
        ]

    return [[]], [list(m) for m in transformation_matches]


def get_constant_and_transformation_indices(
    transformation_smarts: str,
    constant_smarts: str,
    molecule: Molecule,
) -> tuple[list, ...]:
    """Identify constant and trasnformation indices.

    Tries to identify which atoms in a molecule are part of the constant part
    and the transformation part of a transformation.

    Args:
        transformation_smarts (str): SMARTS of fragment that is exchanged
        constant_smarts (str): SMARTS of fragment that is constant
        molecule (Molecule): molecule to match

    Returns:
        tuple[list, ...]: tuple of constant atom indices and transformation atom
        indices. If match is ambiguos, lists of lists or an empty list of list is
        returned.
    """
    if transformation_smarts == "[*:1][H]":
        return get_matches_with_hydrogen(
            constant_smarts=constant_smarts,
            transformation_smarts=transformation_smarts,
            molecule=molecule,
        )
    return get_matches_without_hydrogen(
        constant_smarts=constant_smarts,
        transformation_smarts=transformation_smarts,
        molecule=molecule,
    )


def get_r_group_anchor(
    constant_indices: list,
    transformation_indices: list,
) -> list[int] | tuple[list, ...]:
    """Get atom indices of r group anchors.

    Returns the atom indices of the transformation molecules
    to which the r groups are connected to.

    Returns:
        list[int] | tuple[list, ...]: r-group connection indices.
    """
    if not all(is_match_unique(m) for m in (constant_indices, transformation_indices)):
        return constant_indices, transformation_indices
    intersection = set(constant_indices).intersection(transformation_indices)
    if not intersection:
        raise ValueError(
            f"not intersection\
                \nconstant_indices {constant_indices}\
                \transformation_indices {transformation_indices}",
        )
    return list(intersection)


# -------------------------------------------------------------------------
#
#                       CIRCLE LEVEL OF THINGS
#
# =========================================================================


def try_get_num_atoms_between_rgroups(
    transformation_1: Transformation,
    transformation_2: Transformation,
) -> float:
    """Try get number of atoms between two r groups.

    Tries to determine the distance between two R-Groups that are
    changed respectively in transformation_1 and transformation_2.

    Args:
        transformation_1 (Transformation): First Transformation
        transformation_2 (Transformation): Second Transformation

    Raises:
        NotImplementedError: Thrown if a case i could not think of
            yet is thrown.

    Returns:
        int: -1 if transformation 1 is not unambiguosly defined
             -2 if transformation 2 is not unambiguosly defined
             -3 if both transformations are not unambiguosly defined
            >=0 actual number of bonds between rgroups
    """

    def get_atoms_between(mol: Molecule, index_1: int, index_2: int) -> int:
        return (
            1
            if index_1 == index_2
            else len(Chem.GetShortestPath(mol, index_1, index_2))
        )

    anchor_1, anchor_2 = (
        transformation_1.compound_1_anchor,
        transformation_2.compound_1_anchor,
    )
    molecule = transformation_1.compound_1.rdkit_molecule
    anchor_1_is_unique = isinstance(anchor_1, list)
    anchor_2_is_unique = isinstance(anchor_2, list)
    if not anchor_1_is_unique and anchor_2_is_unique:
        return -1
    if anchor_1_is_unique and not anchor_2_is_unique:
        return -2
    if not all((anchor_1_is_unique, anchor_2_is_unique)):
        return -3

    if len(anchor_1) == len(anchor_2) == 1:
        min_distance = get_atoms_between(
            mol=molecule,
            index_1=anchor_1[0],
            index_2=anchor_2[0],
        )
    else:
        min_distance = float("inf")
        for i in anchor_1:
            for j in anchor_2:
                distance = get_atoms_between(
                    mol=molecule,
                    index_1=i,
                    index_2=j,
                )
                if distance < min_distance:
                    min_distance = distance
    if (
        min_distance == 2
        and len(
            set(transformation_1.compound_1_transformation_indices).intersection(
                set(transformation_2.compound_1_transformation_indices),
            ),
        )
        == 2
    ):
        return 0
    return min_distance


def get_num_atoms_between_rgroups(**kwargs: Transformation) -> int:
    """Calculate number of atoms between r groups in a double transformation cycle.

    following keyword arguments are needed:
        'transformation_1',
        'transformation_2',
        'transformation_3',
        'transformation_4',

    Raises:
        NotImplementedError: thrown if a case occurs that is not implemented yet.

    Returns:
        int: number of atoms between r groups.
    """
    distance = -1
    for pot_transf_1, pot_transf_2 in get_combinations(**kwargs):
        distance = try_get_num_atoms_between_rgroups(
            transformation_1=pot_transf_1,
            transformation_2=pot_transf_2,
        )
        if distance >= 0:
            break
    return int(distance)
