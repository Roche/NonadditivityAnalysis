"""Test nonadditivty.classification.transformation_classification."""

from collections.abc import Callable

import pytest
from rdkit import Chem

from nonadditivity.classification import Compound
from nonadditivity.classification.transformation_classification import (
    _tertiary_nitrogen_generated,
    calculate_fp_similarity,
    get_num_cuts,
    get_num_heavy_atoms_rgroups,
    is_h_replaced,
    num_stereocenters_change,
    stereo_classify_transformation,
    tertiary_amide_generated,
)
from tests._utils import raises_exceptions


def test_is_h_replaced(
    transformation_smarts1: str,
    transformation_smarts2: str,
) -> None:
    """Test nonadditivty.classification.transformation_classification:is_h_replaced.

    Args:
        transformation_smarts1 (str): smarts 1
        transformation_smarts2 (str): smarts 2
    """
    raises_exceptions(
        test_object=transformation_smarts2.split(">>"),
        function=is_h_replaced,
        exception=TypeError,
        transformation_smarts=3,
    )
    with pytest.raises(TypeError):
        is_h_replaced(transformation_smarts=3)
    assert is_h_replaced(
        transformation_smarts=transformation_smarts2.split(">>"),
    )
    assert not is_h_replaced(
        transformation_smarts=transformation_smarts1.split(">>"),
    )


@pytest.mark.parametrize(
    "smarts, goal",
    [
        (["[*:1][*:d]"], 1),
        (["[*:1][*:2]"], 2),
        (["[*:1]asdfcCC"], 1),
    ],
)
def test_get_num_cuts(smarts: list[str], goal: int) -> None:
    """Test nonadditivty.classification.transformation_classification:get_num_cuts.

    Args:
        smarts (list[str]): test input
        goal (int): expected output
    """
    assert get_num_cuts(transformation_smarts=smarts) == goal


@pytest.mark.parametrize(
    "smiles, anchors, solutions",
    [
        (("CCCNCC", "CCCN(CC)CCO"), ([[3], [2]], [3]), (-1, 0)),
        (("CCCNCC", "CCCN(CC)CCO"), ([2], [3]), (0, 0)),
        (("CCCNCC", "CCCN(CC)CCO"), ([3], [3]), (1, 0)),
        (("C1CCNCC1", "OCCN1CCCCC1"), ([3], [3]), (1, 0)),
        (("c1cc[nH]c1", "OCCn1cccc1"), ([3], [3]), (0, 0)),
        (("C1=NCCCN1", "OCCN1C=NCCC1"), ([5], [3]), (2, 0)),
        (("CCC(=O)NC", "CCC(=O)N(C)CCO"), ([4], [4]), (2, 1)),
        (("CCCS(=O)(=O)NC", "CCCS(=O)(=O)N(C)CCO"), ([6], [6]), (1, 0)),
        (("CCCS(=O)(=O)NC", "CCCS(=O)(=O)N(C)CCO"), ([], []), (-1, 0)),
    ],
)
def test_tertiary_nitrogen(
    smiles: tuple[str, str],
    anchors: tuple[list[list[int]] | list[int], list[int]],
    solutions: tuple[int, int],
) -> None:
    """Test ...:tertiary_amide_generated.

    Args:
        smiles (tuple[str, str]): test input smiles
        anchors (tuple[list[list[int]]  |  list[int], list[int]]): test input anchors
        solutions (tuple[int, int]): expected output
    """
    compounds = [Compound(Chem.MolFromSmiles(s), "", "") for s in smiles]
    for func, sol in zip(
        (_tertiary_nitrogen_generated, tertiary_amide_generated),
        solutions,
    ):
        assert (
            func(
                compounds=compounds,
                anchors=anchors,
                transformation_smarts=["[*:1][H]", "[*:1]CCO"],
            )
            == sol
        )
        assert (
            func(
                compounds=compounds,
                anchors=anchors,
                transformation_smarts=["[*:1]C", "[*:]CC"],
            )
            == 0
        )


def test_num_stereocenters_change(
    compound1: Compound,
    compound2: Compound,
    compound3: Compound,
    compound4: Compound,
    transformation_smarts1: str,
    transformation_smarts2: str,
) -> None:
    """Test ...:num_stereocenters_change.

    Args:
        compound1 (Compound): Test input compound 1
        compound2 (Compound): Test input compound 2
        compound3 (Compound): Test input compound 3
        compound4 (Compound): Test input compound 4
        transformation_smarts1 (str): test input smarts
        transformation_smarts2 (str): test input smarts
    """
    raises_exceptions(
        test_object=4,
        function=num_stereocenters_change,
        exception=AttributeError,
        compounds=["wrong_type", 3],
    )
    assert not num_stereocenters_change(
        compounds=[compound1, compound2],
        transformation_smarts=transformation_smarts1,
    )
    assert not num_stereocenters_change(
        compounds=[compound4, compound3],
        transformation_smarts=transformation_smarts1,
    )
    assert num_stereocenters_change(
        compounds=[compound1, compound4],
        transformation_smarts=transformation_smarts2,
    )
    assert num_stereocenters_change(
        compounds=[compound2, compound3],
        transformation_smarts=transformation_smarts2,
    )
    assert num_stereocenters_change(
        compounds=[
            Compound(Chem.MolFromSmiles("C(Cl)(F)Br"), "", ""),
            Compound(Chem.MolFromSmiles("C(Cl)(F)F"), "", ""),
        ],
        transformation_smarts="[*:1]Br>>[*:1]F",
    )


@pytest.mark.parametrize(
    "trans_smart, smiles, sol",
    [
        ("[*:1]CCCl>>[*:1]CCCl", ("CCC[C@@](C)(F)CCCl", "CCC[C@](C)(F)CCCl"), 2),
        ("[*:1]CCCl>>[*:1]CCCl", ("CCC[C@@](C)(F)CCCl", "CCCC(C)(F)CCCl"), 1),
        (
            "[*:1]C[C@@](C)(F)CCCl>>[*:1]C[C@](C)(F)CCCl",
            ("OCC[C@@](C)(F)CCCl", "OCC[C@](C)(F)CCCl"),
            2,
        ),
        (
            "[*:1]C[C@@](C)(F)CCCl>>[*:1]CC(C)(F)CCCl",
            ("OCC[C@@](C)(F)CCCl", "OCCC(C)(F)CCCl"),
            1,
        ),
        ("[*:1]C>>[*:1]Cl", ("CCCC", "CCCCl"), 0),
        ("[*:1][C@H](F)C>>[*:1][C@H](Cl)C", ("CC[C@H](F)C", "CC[C@H](Cl)C"), 0),
        (
            "[*:1][C@@H]1CCCC[C@@H]1[*:2]>>[*:2][C@@H]1CCCC[C@@H]1[*:1]",
            ("CC(O)N[C@H]1CCCC[C@H]1C(N)O", "CC(O)N[C@@H]1CCCC[C@@H]1C(N)O"),
            2,
        ),
        (
            "[*:1][C@@H]1CCCC[C@@H]1[*:2]>>[*:2][C@@H]1CC[C@@H]1[*:1]",
            ("CC(O)N[C@H]1CCCC[C@H]1C(N)O", "CC(O)N[C@@H]1CC[C@@H]1C(N)O"),
            0,
        ),
    ],
)
def test_transformation_stereo(
    trans_smart: str,
    smiles: tuple[str, str],
    sol: int,
) -> None:
    """Test ...:stereo_classify_transformation.

    Args:
        trans_smart (str): test transformation smarts
        smiles (tuple[str,str]): test smiles
        sol (int): expected output
    """
    compounds = [
        Compound(
            Chem.MolFromSmiles(s),
            "",
            "",
        )  # type:ignore pylint:disable=E1101
        for s in smiles
    ]
    assert (
        stereo_classify_transformation(
            transformation_smarts=trans_smart.split(">>"),
            compounds=compounds,
        )
        == sol
    )


@pytest.mark.parametrize(
    "func, sol",
    [
        (get_num_heavy_atoms_rgroups, [3, 6]),
    ],
)
def test_desctiptors(func: Callable[..., list[int]], sol: int) -> None:
    """Test nonadditivty.classification.transformation_classification:rgroupdescriptors.

    Args:
        func (Callable[..., list[int]]): test function
        sol (int): expected output
    """
    smarts = "[*:1]C(=O)Cl>>[*:1]CNC(N)CF"
    c1 = Compound(
        Chem.MolFromSmiles(  # type:ignore pylint:disable=E1101
            "CCC(O)CC(=O)Cl",
        ),
        "",
        "",
    )
    c2 = Compound(
        Chem.MolFromSmiles(  # type:ignore pylint:disable=E1101
            "CCC(O)CCNC(N)CF",
        ),
        "",
        "",
    )
    assert func(compounds=[c1, c2], transformation_smarts=smarts.split(">>")) == sol


def test_calculate_fp_similarity(compound1: Compound, compound2: Compound) -> None:
    """Test ...:calculate_fp_similarity.

    Args:
        compound1 (Compound): test compound 1
        compound2 (Compound): test ccmpound 2
    """
    assert (
        calculate_fp_similarity(compounds=[compound1, compound2]) == 0.855072463768116
    )
    assert (
        calculate_fp_similarity(
            compounds=[
                Compound(
                    Chem.MolFromSmiles(s),  # type:ignore pylint:ignore=E1101
                    "",
                    "",
                )
                for s in ("C", "C")
            ],
        )
        == 1
    )
    assert (
        calculate_fp_similarity(
            compounds=[
                Compound(
                    Chem.MolFromSmiles(s),  # type:ignore pylint:ignore=E1101
                    "",
                    "",
                )
                for s in ("Cl", "F")
            ],
        )
        == 0
    )
