"""Test nonadditivity.classification.rgroup_distance."""
import pytest
from rdkit import Chem

import nonadditivity.classification.rgroup_distance as pm
from nonadditivity.classification import Transformation


@pytest.mark.parametrize(
    "mol, matches, solution",
    [
        (
            "equiv_mol",
            ((0, 1, 2, 3, 4, 5, 6, 8), (0, 1, 2, 3, 5, 6, 7, 8)),
            True,
        ),
        (
            "equiv_mol",
            ((0, 1, 2, 3, 4, 5, 6, 8), (1, 2, 3, 4, 5, 6, 7, 8)),
            False,
        ),
        (
            "equiv_mol",
            ((0, 1, 2, 3, 4, 5, 7), (0, 2, 3, 4, 5, 6, 7), (3,)),
            False,
        ),
        ("equiv_mol", ((0, 1, 2, 3, 4, 5, 7), (0, 1, 3, 4, 5, 7)), False),
        (
            "equiv_mol2",
            ((0, 1, 2, 3, 4, 5, 6, 9), (2, 3, 4, 5, 6, 7, 8, 9)),
            True,
        ),
        (
            "equiv_mol4",
            (
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14),
                (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
            ),
            True,
        ),
    ],
)
def test_all_matches_symetrically_interchangeable(
    mol: str,
    matches: tuple[tuple[int], tuple[int]],
    solution: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test ...:all_matches_symetrically_interchangeable.

    Args:
        mol (str): test input fixture name
        matches (tuple[tuple[int], tuple[int]]): test matches
        solution (bool): expected output
        request (pytest.FixtureRequest): pytest magic.
    """
    assert (
        pm.all_matches_symetrically_interchangeable(
            molecule=request.getfixturevalue(mol),
            matches=matches,
        )
        == solution
    )


@pytest.mark.parametrize(
    "mol, matches, solution",
    [
        ("equiv_mol3", ((0, 1, 2, 3, 4), (0, 1, 2, 3, 5)), True),
        (
            "equiv_mol3",
            ((0, 1, 2, 3, 4), (0, 1, 2, 3, 5), (0, 1, 2, 3, 6)),
            True,
        ),
        (
            "equiv_mol3",
            ((0, 1, 2, 3, 4), (0, 1, 2, 3, 5), (0, 1, 2, 3, 6), (1, 2)),
            False,
        ),
        ("equiv_mol", ((0, 1, 2, 3, 4, 5, 7), (0, 1, 3, 4, 5, 7)), False),
    ],
)
def test_all_matches_chemically_equivalent(
    mol: str,
    matches: tuple[tuple[int], tuple[int]],
    solution: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test ...:all_matches_chemically_equivalent.

    Args:
        mol (str): test input fixture name
        matches (tuple[tuple[int], tuple[int]]): test matches
        solution (bool): expected output
        request (pytest.FixtureRequest): pytest magic.
    """
    assert (
        pm.all_matches_chemically_equivalent(
            molecule=request.getfixturevalue(mol),
            matches=matches,
        )
        == solution
    )


@pytest.mark.parametrize(
    "singlematch, multimatch, numattach, sol",
    [
        (
            [1, 2, 3, 4, 5],
            ((1, 2, 3, 6, 7), (5, 6, 7)),
            1,
            (5, 6, 7),
        ),
        (
            [1, 2, 3, 4, 5],
            ((1, 2, 3, 6, 7), (1, 5, 6, 7)),
            1,
            [],
        ),
    ],
)
def test_get_correct_match(
    singlematch: list[int],
    multimatch: tuple[tuple[int, ...], ...],
    numattach: int,
    sol: tuple[int, ...] | list,
) -> None:
    """Test nonadditivity.classification.rgroup_distance:get_correct_match.

    Args:
        singlematch (list[int]): single match testinput
        multimatch (tuple[tuple[int, ...], ...]): multiple match test input.
        numattach (int): test number of attachment points.
        sol (tuple[int, ...] | list): expected solution.
    """
    assert (
        pm.get_correct_match(
            single_match=singlematch,
            multiple_matches=multimatch,
            num_attachment_points=numattach,
        )
        == sol
    )


@pytest.mark.parametrize(
    "test_val, solution",
    [
        ("CCCC", "CCCC"),
        ("/C=C/CCC", "C=CCCC"),
        ("/[*:1]=C\\CC", "[*:1]=CCC"),
        ("/C1=C/CC", "C1=CCC"),
        ("\\C1=C\\CC", "C1=CCC"),
    ],
)
def test_remove_stereo_at_beginning(test_val: str, solution: str) -> None:
    """Test ...:remove_stereo_at_beginning.

    Args:
        test_val (str): test input
        solution (str): expected output
    """
    assert pm.remove_stereo_at_beginning(smarts=test_val) == solution


@pytest.mark.parametrize(
    "transformation, solution",
    [
        ("transformation1", [list(set(range(22))), [19, 22]]),
        ("equiv_transformation2", [list(range(2, 13)), [0, 1, 2]]),
        (
            "equiv_transformation",
            [
                [
                    list(set(range(2, 15))),
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14],
                ],
                [[0, 1, 2], [10, 11, 12]],
            ],
        ),
    ],
)
def test_get_matches_without_hydrogen(
    transformation: str,
    solution: list[list[int]] | list[list[list[int]]],
    request: pytest.FixtureRequest,
) -> None:
    """Test ...:get_matches_without_hydrogen.

    Args:
        transformation (str): test input transformation
        solution (list[list[int]] | list[list[list[int]]]): expected output
        request (pytest.FixtureRequest): pytest magic.
    """
    transf = request.getfixturevalue(transformation)
    match = pm.get_matches_without_hydrogen(
        constant_smarts=transf.constant_smarts,
        transformation_smarts=transf.transformation_smarts[0],
        molecule=transf.compound_1.rdkit_molecule,
    )
    if any(isinstance(i[0], list) for i in match):
        assert [[list(set(m)) for m in match[i]] for i in (0, 1)] == solution
        return
    assert [list(set(m)) for m in match] == solution


@pytest.mark.parametrize(
    "const, transf, mol, solution",
    [
        ("/C=CCC[*:1]", "[*:1]C", "C=CCCC", [[0, 1, 2, 3], [3, 4]]),
        (
            "[*:1]C1C(F)C(CC)CCC1",
            "[*:1]F",
            "CCC1CCCC(F)C1F",
            [[0, 1, 2, 3, 4, 5, 6, 8, 9], [6, 7]],
        ),
        (
            "[*:2]CC(C)F.[*:3]NCC#N.[*:1]OC",
            "[*:1]Cc1cc([*:2])cnc1[*:3]",
            "COCc1cc(CC(C)F)cnc1NCC#N",
            [
                [0, 1, 6, 7, 8, 9, 13, 14, 15, 16],
                [1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
            ],
        ),
        (
            "[*:1]C(F)(F)CCC",
            "[*:1]F",
            "CCCC(F)(F)F",
            [[0, 1, 2, 3, 4, 5], [3, 6]],
        ),
    ],
)
def test_get_matches_without_hydrogen2(
    const: str,
    transf: str,
    mol: str,
    solution: list[list],
) -> None:
    """Test ...:get_matches_without_hydrogen.

    Args:
        const (str): _description_
        transf (str): _description_
        mol (str): _description_
        solution (list[list]): _description_
    """
    match = pm.get_matches_without_hydrogen(
        constant_smarts=const,
        transformation_smarts=transf,
        molecule=Chem.MolFromSmiles(mol),
    )
    if any(isinstance(i[0], list) for i in match):
        assert [[list(set(m)) for m in match[i]] for i in (0, 1)] == solution
        return
    assert [list(set(m)) for m in match] == solution


def test_get_matches_with_hydrogen(equiv_transformation2: Transformation) -> None:
    """Test nonadditivity.classification.rgroup_distance:get_matches_with_hydrogen.

    Args:
        equiv_transformation2 (Transformation): test input.
    """
    transformation = equiv_transformation2.get_reverse()
    match = pm.get_matches_with_hydrogen(
        constant_smarts=transformation.constant_smarts,
        transformation_smarts=transformation.transformation_smarts[0],
        molecule=transformation.compound_1.rdkit_molecule,
    )
    assert [list(set(v)) for v in match[0]] == [
        [*list(range(11)), 14],
        [*list(range(11)), 24],
    ]
    assert [list(set(v)) for v in match[1]] == [
        list(set(v))
        for v in [
            [0, 11],
            [0, 12],
            [0, 13],
            [2, 14],
            [3, 15],
            [5, 16],
            [5, 17],
            [6, 18],
            [7, 19],
            [7, 20],
            [7, 21],
            [8, 22],
            [9, 23],
            [10, 24],
        ]
    ]


@pytest.mark.parametrize(
    "ci, ti, goal",
    [
        ([], [], ([], [])),
        ([[]], [[]], ([[]], [[]])),
        ([1, 2, 3, 4], [3, 4, 5, 6], [3, 4]),
        ([1, 2, 3, 4], [], ([1, 2, 3, 4], [])),
    ],
)
def test_transformation_anchors(ci: list, ti: list, goal: list) -> None:
    """Test nonadditivity.classification.rgroup_distance:get_r_group_anchor.

    Args:
        ci (list): constant test indices
        ti (list): transformation test indices
        goal (list): expected output
    """
    assert (pm.get_r_group_anchor(ci, ti)) == goal


@pytest.mark.parametrize(
    "t1,t2,t3,t4, sol",
    [
        (
            "transformation1",
            "transformation2",
            "transformation3",
            "transformation4",
            5,
        ),
        (
            "transformation5",
            "transformation6",
            "transformation7",
            "transformation8",
            6,
        ),
    ],
)
def test_get_distance_between_r_groups(
    t1: str,
    t2: str,
    t3: str,
    t4: str,
    sol: int,
    request: pytest.FixtureRequest,
) -> None:
    """Test ...:get_num_atoms_between_rgroups.

    Args:
        t1 (str): test transformation 1
        t2 (str): test transformation 2
        t3 (str): test transformation 3
        t4 (str): test transformation 4
        sol (int): expected solution
        request (pytest.FixtureRequest): pytest magic
    """
    assert (
        pm.get_num_atoms_between_rgroups(
            transformation_1=request.getfixturevalue(t1),
            transformation_2=request.getfixturevalue(t2),
            transformation_3=request.getfixturevalue(t3),
            transformation_4=request.getfixturevalue(t4),
        )
        == sol
    )
