"""Test nonadditivity.classification.circle_classification."""
from collections.abc import Callable

import pytest
from rdkit import Chem

from nonadditivity.classification import Transformation
from nonadditivity.classification.circle_classification import (
    _substituents_in_same_ring_system,
    circle_has_stereoinversion_in_transformation,
    get_chi0_diff,
    get_chi1_diff,
    get_chi2_diff,
    get_compound_stereocenter_classification,
    get_formal_charge_diff,
    get_hbond_acceptor_diff,
    get_hbond_donor_diff,
    get_log_p_diff,
    get_max_num_heavy_atom_in_transformation,
    get_max_num_mmpdb_cuts,
    get_min_transformation_tanimoto,
    get_num_heavy_atoms_diff,
    get_num_rot_bonds_diff,
    get_scaffold_smiles,
    get_sp3_carbon_diff,
    get_tertiary_amide_formed,
    get_tpsa_diff,
    substituents_in_same_ring_system,
)
from nonadditivity.classification.compound_classification import (
    get_aromatic_indices,
    get_fused_ring_indices,
)

big_ring = (
    "Cc1cc2cc(C)cnc2cc1CC(O)CCc1cc2cccnc2cc1CC1CCCC2CCCCC2CC1CC1Cc2ccccc2CC1CC1="
    "CC=CC2=C1C3=CC5=C4C(=C23)C=CC=C4CCC5"
)


@pytest.mark.parametrize(
    "func, sol",
    [
        (get_log_p_diff, 0.3460000000000001),
        (get_num_heavy_atoms_diff, 4),
        (get_sp3_carbon_diff, 3),
        (get_hbond_donor_diff, 1),
        (get_formal_charge_diff, 0.0),
        (get_num_rot_bonds_diff, 2),
        (get_hbond_acceptor_diff, 1),
        (get_tpsa_diff, 20.230000000000004),
        (get_chi0_diff, 3.1547005383792524),
        (get_chi1_diff, 1.8045304526403108),
        (get_chi2_diff, 1.3427031358353583),
    ],
)
def test_rdkit_deltas(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
    func: Callable[..., float],
    sol: float,
) -> None:
    """Test rdkit descriptors.

    Args:
        transformation1 (Transformation): Transformation 1
        transformation2 (Transformation): Transformation 2
        transformation3 (Transformation): Transformation 3
        transformation4 (Transformation): Transformation 4
        func (Callable[..., float]): function to test
        sol (float): expected return
    """
    assert (
        func(
            transformation_1=transformation1,
            transformation_2=transformation2,
            transformation_3=transformation3,
            transformation_4=transformation4,
        )
        == sol
    )


@pytest.mark.parametrize(
    "func, sol",
    [
        (get_compound_stereocenter_classification, "Unassigned"),
        (circle_has_stereoinversion_in_transformation, "None"),
        (get_tertiary_amide_formed, "False"),
        (get_max_num_heavy_atom_in_transformation, 4),
        (get_max_num_mmpdb_cuts, 1),
    ],
)
def test_max_in_circle_transformation(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
    func: Callable[..., str] | Callable[..., int],
    sol: int | str,
) -> None:
    """Test functions that apply transformation property to circle.

    Args:
        transformation1 (Transformation): Transformation 1
        transformation2 (Transformation): Transformation 2
        transformation3 (Transformation): Transformation 3
        transformation4 (Transformation): Transformation 4
        func (Callable[..., str] | Callable[..., int]): function to test
        sol (int | str): expected return
    """
    assert (
        func(
            transformation_1=transformation1,
            transformation_2=transformation2,
            transformation_3=transformation3,
            transformation_4=transformation4,
        )
        == sol
    )


def test_tanimoto(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
) -> None:
    """Test get_min_transformation_tanimoto.

    Args:
        transformation1 (Transformation): Transformation 1
        transformation2 (Transformation): Transformation 2
        transformation3 (Transformation): Transformation 3
        transformation4 (Transformation): Transformation 4
    """
    assert (
        get_min_transformation_tanimoto(
            transformation_1=transformation1,
            transformation_2=transformation2,
            transformation_3=transformation3,
            transformation_4=transformation4,
        )
        == 0.6626506024096386
    )


def test_get_scaffold(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
) -> None:
    """Test get scaffold.

    Args:
        transformation1 (Transformation): Transformation 1
        transformation2 (Transformation): Transformation 2
        transformation3 (Transformation): Transformation 3
        transformation4 (Transformation): Transformation 4
    """
    assert (
        get_scaffold_smiles(
            transformation_1=transformation1,
            transformation_2=transformation2,
            transformation_3=transformation3,
            transformation_4=transformation4,
        )
        == "CCCc1occc1Cc1cc(C[C@@](C)(O)[*])c(CCC)c([*])c1"
    )


@pytest.mark.parametrize(
    "smiles, idx1, idx2, solution",
    [
        ("Cc1ccccc1CC(O)C", 1, 6, True),
        ("Cc1ccccc1CC(O)C", 1, 7, False),
        ("Cc1cocc1CC(O)C", 1, 5, True),
        ("Cc1cocc1CC(O)C", 1, 6, False),
        ("Cc1cc2cc(C)cnc2cc1CC(O)C", 1, 11, True),
        ("Cc1cc2cc(C)cnc2cc1CC(O)C", 1, 12, False),
        ("Cc1cc2cc(C)cnc2cc1CC(O)C", 1, 5, True),
        ("Cc1ncnc(CC)c1c1cccc(C)c1CC(O)C", 13, 15, True),
        ("Cc1ncnc(CC)c1c1cccc(C)c1CC(O)C", 13, 5, False),
        (big_ring, 5, 1, True),
        (big_ring, 5, 15, False),
        (big_ring, 44, 42, False),
        (big_ring, 52, 66, True),
        (big_ring, 52, 69, False),
    ],
)
def test_substituents_in_same_ring_system(
    smiles: str,
    idx1: int,
    idx2: int,
    solution: bool,
) -> None:
    """Test _substituents_in_same_ring_system.

    Args:
        smiles (str): test smiles for molecule
        idx1 (int): test anchor 1
        idx2 (int): test anchor 2
        solution (bool): expected ouptut
    """
    mol = Chem.MolFromSmiles(smiles)
    aromatic_indices = get_aromatic_indices(molecule=mol)
    fused_rings = get_fused_ring_indices(molecule=mol)
    assert (
        _substituents_in_same_ring_system(
            molecule=mol,
            index_1=idx1,
            index_2=idx2,
            aromatic_indices=aromatic_indices,
            fused_rings=fused_rings,
        )
        == solution
    )


def test_substituent_in_same_ring_system(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
) -> None:
    """Test substituents_in_same_ring_system.

    Args:
        transformation1 (Transformation): test transformation1 input
        transformation2 (Transformation): test transformation2 input
        transformation3 (Transformation): test transformation3 input
        transformation4 (Transformation): test transformation4 input
    """
    assert not substituents_in_same_ring_system(
        transformation_1=transformation1,
        transformation_2=transformation2,
        transformation_3=transformation3,
        transformation_4=transformation4,
    )
