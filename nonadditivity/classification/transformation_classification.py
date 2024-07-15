"""Implements transformation classification.

All these functions are higly specific, do not use them outside
of their intended application.
"""

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from rdkit import Chem, DataStructs

from nonadditivity.classification.utils import is_unique
from nonadditivity.utils.types import Atom, Molecule

if TYPE_CHECKING:
    from nonadditivity.classification import Compound

AMIDE_PATTERN = Chem.MolFromSmarts(  # type:ignore pylint:disable=E1101
    "[NX3][CX3](=[OX1])[#6]",
)


def get_num_cuts(**kwargs: str) -> int:
    """Returns number of mmpdb cuts made in a transformation.

    use keyword argument 'transformation_smarts' for this function to work.


    Returns:
        int: number of mmpdb cuts
    """
    smarts: str = kwargs["transformation_smarts"][0]
    num = 0
    for i in range(3):
        if f"[*:{i+1!s}]" in smarts:
            num += 1
    return num


def is_h_replaced(**kwargs: str) -> bool:
    """Check whether one side of the transformation is only a hydrogen.

    Returns true if an H is replaced or something is replaced by
    a H in this transformation.

    use keyword argumnt 'transformation_smarts' for it to work.

    Returns:
        bool: True if H is part of transformation
    """
    return "[*:1][H]" in kwargs["transformation_smarts"]


def _tertiary_nitrogen_generated(**kwargs) -> int:
    """Check whether a tertiary nitrogen is formed in transformation.

    use keyword arguments 'compounds' and 'anchors' for this function to work.

    Returns:
        int: 0 == No, 1 == Yes, -1 == Not identifiable, 2 == yes and in both cases
        the nitrogen is conjugated..
    """

    def is_conjugated(atom: Atom) -> bool:
        return any(b.GetIsConjugated() for b in atom.GetBonds())

    if not is_h_replaced(**kwargs):
        return 0
    molecule_1, molecule_2 = (cpd.rdkit_molecule for cpd in kwargs["compounds"])
    anchor_1_idxs, anchor_2_idxs = kwargs["anchors"]
    for val in (-1, 0):
        if any(is_unique(a) == val for a in (anchor_1_idxs, anchor_2_idxs)):
            return val
    anchor_1_atom, anchor_2_atom = (
        molecule_1.GetAtomWithIdx(
            anchor_1_idxs[0],
        ),
        molecule_2.GetAtomWithIdx(
            anchor_2_idxs[0],
        ),
    )
    if (
        any(a.GetIsAromatic() for a in (anchor_1_atom, anchor_2_atom))
        or anchor_1_atom.GetSymbol() == anchor_2_atom.GetSymbol() != "N"
        or {a.GetTotalNumHs() for a in [anchor_1_atom, anchor_2_atom]} != {0, 1}
    ):
        return 0
    if any(is_conjugated(a) for a in (anchor_1_atom, anchor_2_atom)):
        return 2
    return 1


def tertiary_amide_generated(**kwargs) -> int:
    """Check Whether tertiary amide is formed in transformation.

    use keyword arguments 'compounds' and 'anchors' for this function to work.

    Returns:
        int: 0 == No | Not identifiable, 1 == Yes.
    """

    def anchor_is_in_pattern(pattern: Molecule, **kwargs) -> bool:
        molecule = kwargs["compounds"][0].rdkit_molecule
        anchor = kwargs["anchors"][0][0]
        potential_matches = molecule.GetSubstructMatches(pattern)
        return any(anchor in m for m in potential_matches)

    tertiary_nitrogen = _tertiary_nitrogen_generated(
        **kwargs,
    )
    if tertiary_nitrogen < 2:
        return 0
    return int(anchor_is_in_pattern(pattern=AMIDE_PATTERN, **kwargs))


def stereo_classify_transformation(**kwargs) -> int:
    """Check whether transforamtion is only changing stereochemistry.

    Checks whether a transformation is only a stereoinversion or going
    from a compound with assigned stereochemistry to the same compound
    without assigned stereocenters.

    use keyword arguments 'compounds'
    for this function to work.

    Returns:
        int: 0 == Transformation not stereo only
            1 == assigned -> unassigned
                2 == stereoinversion
    """
    molecules: list[Molecule] = [c.rdkit_molecule for c in kwargs["compounds"]]

    # if not same number of atoms, change has to be other than stereochemistry!
    if molecules[0].GetNumAtoms() != molecules[1].GetNumAtoms():
        return 0
    # if we can match without stereochemistry, the transformation has to be
    # stereo related
    if (molecules[0].GetSubstructMatch(molecules[1], useChirality=False)) and (
        molecules[1].GetSubstructMatch(molecules[0], useChirality=False)
    ):
        # Unassigned stereocenters are matched with assigned stereocenters
        # --> we go from assigned to unassigned or vice versa
        if molecules[0].GetSubstructMatch(molecules[1], useChirality=True):
            return 1
        # compounds are matched without chirality but not with chirality,
        # --> change is only stereochem related!
        return 2
    # if we cannot match the two hole structures, transformation is not
    # stereochemistry related.
    return 0


def _get_phys_chem_property(
    function: Callable[[Molecule], float],
    **kwargs,
) -> list[float]:
    """Apply function to both transformation parts.

    Takes the transformation smarts, converts them to smiles,
    and then lets the function caluclate a value for the molecules
    from the just generated smiles.

    use keyword argument 'compounds' and 'transformation_smarts'
    for this function to work.

    Args:
        function (Callable[[Molecule], float]): rdkit description function.
        add_hs (bool, optional): Whether to add Hydrogens. Defaults to False.
        **kwargs: kwargs used for this function.

    Returns:
        list[float]: values calculated.
    """
    mols = [
        Chem.MolFromSmarts(c)  # type: ignore pylint: disable=E1101
        for c in kwargs["transformation_smarts"]
    ]
    for mol in mols:
        with contextlib.suppress(Exception):
            Chem.SanitizeMol(mol)  # type: ignore pylint: disable=E1101
    return [function(mol) for mol in mols]


def get_num_heavy_atoms_rgroups(**kwargs: str) -> list[int]:
    """Calculate number of heavy atoms in the rgroups of a transformation.

    use keyword argument 'transformation_smarts' for this function to work.

    Returns:
        int: difference in heavy atoms
    """

    def get_num_heavy(molecule: Molecule) -> int:
        return molecule.GetNumHeavyAtoms()

    return [int(x) for x in _get_phys_chem_property(function=get_num_heavy, **kwargs)]


def calculate_fp_similarity(**kwargs) -> float:
    """Calculate Tanimoto of Morgan fingerprints (2) of compounds in transformation.

    use keyword argument 'compounds' for this function to work.

    Returns:
        float: Tanimoto of Fingerprints
    """
    compounds: list[Compound] = kwargs["compounds"]
    return DataStructs.TanimotoSimilarity(  # type: ignore pylint: disable=E1101
        *[m.get_property(m.Properties.MORGANFP) for m in compounds],
    )
