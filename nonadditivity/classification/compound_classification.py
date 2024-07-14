"""Implement compound classification.

Implements all classification functions on a compound level
that are called in the Compound class.
All these functions are higly specific, do not use them outside
of their intended application.
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, GraphDescriptors, rdMolDescriptors

from nonadditivity.utils.types import Molecule


def get_aromatic_indices(**kwargs: Molecule) -> set[int]:
    """Get list of indices for all aromatic atoms in molecule.

    use keyword argument 'molecule' for this function to work.

    Returns:
        set[int]: indices of aromatic atoms in molecule
    """
    mol = kwargs["molecule"]
    return {
        i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic()
    }


def get_fused_ring_indices(**kwargs: Molecule) -> list[set[int]]:
    """Detects the number of fused rings in the given molecule.

    Returns:
        list[set[int]]: list of sets with fused ring system indices.
    """
    aromatic_atoms = get_aromatic_indices(**kwargs)
    sssr = Chem.GetSSSR(kwargs["molecule"])
    fused_rings = []
    for outer_i, outer in enumerate(sssr):
        if not all(i in aromatic_atoms for i in outer):
            continue
        newring = set(outer)
        same_length = False
        while not same_length:
            old_length = len(newring)
            for _, inner in enumerate(sssr[outer_i + 1 :]):
                if not all(i in aromatic_atoms for i in inner):
                    continue
                if len(newring.intersection(inner)) >= 2:
                    newring.update(inner)
            same_length = old_length == len(newring)
        if newring == set(outer):
            continue
        if not any(len(i.intersection(newring)) == len(newring) for i in fused_rings):
            fused_rings.append(newring)
    return fused_rings


def get_num_heavy_atoms(**kwargs: Molecule) -> int:
    """Get number of heavy atoms in a molecule.

    Returns number of heavy atoms of the compound
    using rdkits Mol.GetNumHeavyAtoms()

    use keyword argument 'molecule' for this function to work.


    Returns:
        int: Number of heavy atoms.
    """
    molecule = kwargs["molecule"]
    return molecule.GetNumHeavyAtoms()


def get_num_stereo_centers(**kwargs: Molecule) -> int:
    """Get number of stereocenter in a molecule.

    Returns number of stereo Centers of the compound
    using rdkits new implementation of
    Chem.FindMolChiralCenters()

    use keyword argument 'molecule' for this function to work.


    Returns:
        int: Number of stereo centers.
    """
    molecule: Molecule = kwargs["molecule"]
    return len(
        Chem.FindMolChiralCenters(
            mol=molecule,
            includeUnassigned=True,
            useLegacyImplementation=False,
        ),
    )


def has_unassigned_stereocenters(**kwargs: Molecule) -> bool:
    """Check if a molecule has unassigned stereocenters.

    use keyword argument 'molecule' for this function to work.

    Returns:
        bool: True if mol has unassigned stereocenter
    """
    molecule: Molecule = kwargs["molecule"]
    centers: list[str] = [
        v[1]
        for v in Chem.FindMolChiralCenters(
            mol=molecule,
            includeUnassigned=True,
        )
    ]
    return any(c.lower() == "?" for c in centers)


def get_num_h_bond_acceptors(**kwargs: Molecule) -> float:
    """Calculate number of hbond acceptors for the molecule.

    use keyword argument 'molecule' for this function to work.

    Returns:
        float: numn h bond acceptors
    """
    return rdMolDescriptors.CalcNumHBA(  # pylint: disable=I1101
        kwargs["molecule"],
    )


def get_num_h_bond_donors(**kwargs: Molecule) -> float:
    """Calculate number of hbond donors for the molecule.

    use keyword argument 'molecule' for this function to work.

    Returns:
        float: numn h bond donors
    """
    return rdMolDescriptors.CalcNumHBD(  # pylint: disable=I1101
        kwargs["molecule"],
    )


def get_charge(**kwargs: Molecule) -> int:
    """Calculate charge of a molecule.

    use keyword argument 'molecule' for this function to work.

    Returns:
        float: charge
    """
    return Chem.GetFormalCharge(  # type: ignore pylint: disable=E1101
        kwargs["molecule"],
    )


def get_polar_surface_area(**kwargs: Molecule) -> float:
    """Calculate the topological polar surface areas for a molecule.

    use keyword argument 'molecule' for this function to work.

    Returns:
        float: polar surface area
    """
    return rdMolDescriptors.CalcTPSA(kwargs["molecule"])  # pylint: disable=I1101


def get_log_p(**kwargs: Molecule) -> list[float]:
    """Calculate log_p for the molecule.

    use keyword argument 'molecule' for this function to work.

    Returns:
        list[float]: log_p values
    """
    return Crippen.MolLogP(kwargs["molecule"])


def get_num_sp3_carbons(**kwargs: Molecule) -> int:
    """Calculate number of sp3 carbons in molecule.

    use keyword argument 'molecule' for this function to work.

    Returns:
        list[float]: fraction of sp3 carbons
    """
    return sum(
        a.GetHybridization() == Chem.HybridizationType.SP3 and a.GetSymbol() == "C"  # pylint: disbale=E1101
        for a in kwargs["molecule"].GetAtoms()
    )


def get_num_rotatable_bonds(**kwargs: Molecule) -> int:
    """Calculate number of rotatable bonds for a molecule transformation.

    use keyword argument 'molecule' for this function to work.

    Returns:
        list[float]: num rotatable bonds
    """
    # molecule = Chem.AddHs(kwargs["molecule"])
    return rdMolDescriptors.CalcNumRotatableBonds(
        kwargs["molecule"],
    )  # pylint: disable=I1101


def get_chi0(**kwargs: Molecule) -> list[float]:
    """Calculate chi0 for both parts of the transformation.

    use keyword argument 'molecule' for this function to work.

    Returns:
        list[float]: Chi0 values
    """
    return GraphDescriptors.Chi0(kwargs["molecule"])


def get_chi1(**kwargs: Molecule) -> list[float]:
    """Calculate chi1 for both parts of the transformation.

    use keyword argument 'molecule' for this function to work.

    Returns:
        list[float]: chi1 values
    """
    return GraphDescriptors.Chi1(kwargs["molecule"])


def get_chi2(**kwargs: Molecule) -> list[float]:
    """Calculates chi2 for both parts of the transformation.

    use keyword argument 'molecule' for this function to work.

    Returns:
        list[float]: chi2 values
    """
    return GraphDescriptors.Chi2n(kwargs["molecule"])


def get_morgan_fp(
    **kwargs: Molecule,
) -> DataStructs.cDataStructs.UIntSparseIntVect:  # pylint: disable=I1101
    """Calculate the morgan fingerprint with radius 2 for a molecule.

    use keyword argument 'molecule' for this function to work.


    Returns:
        DataStructs.cDataStructs.UIntSparseIntVect: MorganFP with radius 2
    """
    return rdMolDescriptors.GetMorganFingerprint(  # type: ignore pylint:disable=E1101
        mol=kwargs["molecule"],
        radius=2,
    )
