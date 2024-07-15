"""Test nonadditivity.classification.compound_classification."""

from collections.abc import Callable

import pytest

from nonadditivity.classification.compound_classification import (
    get_aromatic_indices,
    get_charge,
    get_chi0,
    get_chi1,
    get_chi2,
    get_fused_ring_indices,
    get_log_p,
    get_morgan_fp,
    get_num_h_bond_acceptors,
    get_num_h_bond_donors,
    get_num_heavy_atoms,
    get_num_rotatable_bonds,
    get_num_sp3_carbons,
    get_num_stereo_centers,
    get_polar_surface_area,
    has_unassigned_stereocenters,
)
from nonadditivity.utils.types import Molecule
from tests._utils import raises_exceptions


@pytest.mark.parametrize(
    "func, exc, sol",
    [
        (get_num_heavy_atoms, AttributeError, 23),
        (get_num_stereo_centers, Exception, 1),
        (get_charge, Exception, 0),
        (get_num_h_bond_acceptors, Exception, 2),
        (get_num_h_bond_donors, Exception, 1),
        (get_polar_surface_area, Exception, 33.370000000000005),
        (get_log_p, Exception, 5.265200000000005),
        (get_num_sp3_carbons, Exception, 10),
        (get_num_rotatable_bonds, Exception, 8),
        (get_chi0, Exception, 16.872032720186702),
        (get_chi1, Exception, 10.92516481473553),
        (get_chi2, Exception, 6.468033721680204),
        (get_aromatic_indices, Exception, {3, 4, 5, 6, 8, 9, 10, 11, 12, 16, 17}),
        (get_fused_ring_indices, Exception, []),
    ],
)
def test_rdkit_wrappers(
    compound_molecule1: Molecule,
    func: Callable[..., float],
    exc: Exception,
    sol: float,
) -> None:
    """Testing rdkit wrappers on compound level.

    Args:
        compound_molecule1 (Molecule):  molecule
        func (Callable[..., float]): tets function
        exc (Exception): exception raised when wrong type
        sol (float): expected solution.
    """
    raises_exceptions(
        test_object=compound_molecule1,
        function=func,
        exception=exc,
        molecule="wrongtype",
    )
    assert func(molecule=compound_molecule1) == sol


def test_morganfp(compound_molecule1: Molecule) -> None:
    """Test get_morgan_fp.

    Args:
        compound_molecule1 (Molecule): molecule.
    """
    raises_exceptions(
        test_object=compound_molecule1,
        function=get_morgan_fp,
        exception=Exception,
        molecule="wrongtype",
    )
    result = get_morgan_fp(molecule=compound_molecule1)
    assert result.GetTotalVal() == 64
    assert result.GetLength() == 4294967295
    assert result.GetNonzeroElements() == {
        3624155: 2,
        28274933: 1,
        73745272: 1,
        170894115: 1,
        242694095: 1,
        632776244: 1,
        864662311: 1,
        878182647: 1,
        951226070: 3,
        992462230: 1,
        994485099: 1,
        1016841875: 1,
        1038762601: 1,
        1057480134: 1,
        1173125914: 2,
        1267407792: 1,
        1396348621: 1,
        1429883190: 1,
        1542631284: 1,
        1683958275: 1,
        1902789874: 1,
        2064305509: 1,
        2245277810: 1,
        2245384272: 6,
        2246728737: 3,
        2292369753: 1,
        2315485585: 1,
        2444880134: 1,
        2787503806: 2,
        2947832943: 1,
        3143719699: 1,
        3189457552: 1,
        3217380708: 5,
        3218693969: 5,
        3435725979: 1,
        3466404006: 1,
        3537123720: 1,
        3542456614: 2,
        4089138501: 3,
        4121755354: 1,
        4194776273: 1,
    }


@pytest.mark.parametrize(
    "mol, unassigned",
    [
        ("no_stereo_mol", False),
        ("compound_molecule1", False),
        ("compound_molecule2", False),
        ("unassigned_stereo_mol", True),
    ],
)
def test_stereochemistry(
    mol: str,
    unassigned: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test stereochemistry assignment on compound level.

    Args:
        mol (str): molecule fixture name
        unassigned (bool): whether has mol has unassigned stereocenters.
        request (pytest.FixtureRequest): pytest magic
    """
    molecule: Molecule = request.getfixturevalue(mol)
    assert has_unassigned_stereocenters(molecule=molecule) == unassigned
