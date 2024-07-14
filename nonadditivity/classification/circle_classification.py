"""Implement circle classification.

Implements all the classification functions called in classification_classes:Circle
that are not responsible for the ortho or the r group distance classification.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from nonadditivity.classification.utils import is_unique

if TYPE_CHECKING:
    from nonadditivity.classification import (
        Compound,
        Transformation,
    )
    from nonadditivity.utils.types import Molecule


AMINE_DICT = {-1: "Not identifiable", 0: "False", 1: "True"}
STEREO_DICT = {0: "None", 1: "unassigned", 2: "inversion"}


def _any_compound_has_property(
    property_key: Compound.Properties,
    **kwargs: Transformation,
) -> bool:
    """Return true if any of the compounds have true for <property_key>.

    kwargs 'transformation_1', 'transformation_2', 'transformation_3' and
    'transformation_4' needed for this to work.

    Args:
        property_key (Compound.Properties): property to look at.
        kwargs: keyword arguments.

    Returns:
        bool: True if any any of the compounds have True at <property_key>
    """
    compound_1 = kwargs["transformation_1"].compound_1
    compound_2 = kwargs["transformation_1"].compound_2
    compound_3 = kwargs["transformation_4"].compound_2
    compound_4 = kwargs["transformation_4"].compound_1
    return any(
        c.get_property(property_key)
        for c in (compound_1, compound_2, compound_3, compound_4)
    )


def _get_max_in_circle_transformations(
    property_key: Transformation.Properties,
    iterators: tuple[int, ...],
    **kwargs: Transformation,
) -> int:
    """Get for maximum value of property for transformations in iterators.

    Returns maximum value for a <property_key> for any of the transformations
    given in <iterators> for a given circle.

    Args:
        property_key (Transformation.Properties): property to get max value for.
        iterators (tuple[int]): which transformations to look at.
        kwargs: kwargs to use in this.

    Returns:
        int: max value found in <iterators> transformations for <property_key>
    """
    max_val = [
        t.get_property(property_key)
        for t in [kwargs[f"transformation_{i}"] for i in iterators]
    ]
    if isinstance(max_val[0], Iterable):
        max_val = list(np.concatenate(max_val))
    return max(max_val)


def get_compound_stereocenter_classification(**kwargs: Transformation) -> str:
    """Get stereo classification for compounds in circle.

    use kwargs transformation_1, 2, 3 and 4 for this to work.

    Returns:
        str: stereo classification for compounds
    """
    prop = kwargs["transformation_1"].compound_1.Properties
    if not _any_compound_has_property(prop.NUM_STERO_CENTERS, **kwargs):
        return "None"
    return (
        "Unassigned"
        if _any_compound_has_property(prop.HAS_UNASSIGNED_STEREOCENTERS, **kwargs)
        else "Assigned"
    )


def circle_has_stereoinversion_in_transformation(**kwargs: Transformation) -> str:
    """Check whether a transformation in the circle is a stereoinversion.

    Checks whether any transformation in a circle is only a stereoinversion,
    resp. a change from assigned to unassigned stereochemistry.

    use kwargs transformation_1 and transformation_2 for this to work.

    Returns:
        str: 'None' == transformations are not stereochemistry related
            'unassigned' == at least one transformation transforms an assigned
            stereo isomer to an unassigned stereoisomer.
            'inversion' == at least one transformation is a stereoinversion
    """
    prop = kwargs["transformation_1"].Properties.TRANSFORMATION_STEREO
    return STEREO_DICT[
        _get_max_in_circle_transformations(
            property_key=prop,
            iterators=(1, 2),
            **kwargs,
        )
    ]


def get_tertiary_amide_formed(**kwargs: Transformation) -> str:
    """Check whether any transformation in a circle creates a tertiary amide.

    use kwargs transformation_1, 2, 3 and 4 for this to work.

    Returns:
        str: 'True' == tertiary amide formed in any of the transformations
            'False' == no tertiary amide formed in any of the transformations
            'Not Identifiable' == not able to perform check.
    """
    prop = kwargs["transformation_1"].Properties.TERTIARY_AMIDE_FORMED
    return AMINE_DICT[
        _get_max_in_circle_transformations(
            property_key=prop,
            iterators=(1, 2, 3, 4),
            **kwargs,
        )
    ]


def get_max_num_heavy_atom_in_transformation(**kwargs: Transformation) -> int:
    """Return biggest transformation of a circle.

    use kwargs transformation_1 and transformation_2 for this to work.

    Returns:
        int: highest number of atoms found in any r group.
    """
    prop = kwargs["transformation_1"].Properties.NUM_HEAVY_ATOMS_IN_RGROUPS
    return _get_max_in_circle_transformations(
        property_key=prop,
        iterators=(1, 2),
        **kwargs,
    )


def get_max_num_mmpdb_cuts(**kwargs: Transformation) -> int:
    """Returns highest number of mmpdb cuts in a circle.

    use kwargs transformation_1 and transformation_2 for this to work.

    Returns:
        int: highest num mmpdb cuts in a circle.
    """
    prop = kwargs["transformation_1"].Properties.NUM_MMPDB_CUTS
    return _get_max_in_circle_transformations(
        property_key=prop,
        iterators=(1, 2),
        **kwargs,
    )


def get_min_transformation_tanimoto(**kwargs: Transformation) -> float:
    """Get smaller tanimoto of fingerprints for either t1 and t4 or t2 and t3.

    use kwargs transformation_1, 2, 3 and for this method to work.

    Returns:
        float: mean of tanimoto of fingerprints of transformation1 and 4
    """
    t1 = kwargs["transformation_1"]
    t4 = kwargs["transformation_4"]
    t3 = kwargs["transformation_3"]
    t2 = kwargs["transformation_2"]
    return min(
        np.mean(
            [
                t2.get_property(t2.Properties.MFP2_SIMILARITY),
                t3.get_property(t3.Properties.MFP2_SIMILARITY),
            ],
        ),
        np.mean(
            [
                t1.get_property(t1.Properties.MFP2_SIMILARITY),
                t4.get_property(t4.Properties.MFP2_SIMILARITY),
            ],
        ),
    )


def __calc_max_delta(
    val_1: float,
    val_2: float,
    val_3: float,
    val_4: float,
) -> float:
    """Get biggest absolute difference in (val1-val3) and (val2 - val4).

    Args:
        val_1 (float): val1
        val_2 (float): val2
        val_3 (float): val3
        val_4 (float): val4

    Returns:
        float: max(abs(v1-v3),abs(v2-v4))
    """
    return max(
        abs(
            val_1 - val_3,
        ),
        abs(
            val_2 - val_4,
        ),
    )


def _get_max_diagonal_delta_from_compounds(
    property_key: Compound.Properties,
    **kwargs: Transformation,
) -> float:
    """Find the biggest delta in property in a double transformation cycle.

    use kwargs transformation_1, 2, 3 and 4 for this to work.

    Args:
        property_key (Transformation.Properties): name of the transformation property
        kwargs: kwargs needed for this function.

    Returns:
        float: max(abs(diff_1,2)) for diff_1,2 being the two diagonal differencees.
    """
    val_1 = kwargs["transformation_1"].compound_1.get_property(property_key)
    val_2 = kwargs["transformation_1"].compound_2.get_property(property_key)
    val_3 = kwargs["transformation_4"].compound_2.get_property(property_key)
    val_4 = kwargs["transformation_4"].compound_1.get_property(property_key)

    return __calc_max_delta(val_1, val_2, val_3, val_4)


def get_hbond_donor_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in h bond donors for double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in h bond donors
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.NUM_HBD
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_hbond_acceptor_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in h bond acceptors for double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in h bond acceptors
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.NUM_HBA
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_formal_charge_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in formal charge for double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in formal charge
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.CHARGE
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_tpsa_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in tpsa for double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in tpsa
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.TPSA
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_num_rot_bonds_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in num of rotatable bonds for double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in number of rotatable bonds
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.NUM_ROT_BONDS
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_sp3_carbon_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in num of spc 3 carbons for double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in spc 3 carbon fraction
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.NUM_SP3_CARBONS
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_log_p_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in log p for double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in log p
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.LOGP
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_num_heavy_atoms_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in heavy atoms double transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in heavy atoms
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.NUM_HEAVY_ATOMS
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_chi0_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in chi1 in the 4 r groups in  transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in chi1
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.CHI0
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_chi1_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in chi1 in the 4 r groups in  transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in chi1
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.CHI1
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_chi2_diff(**kwargs: Transformation) -> float:
    """Get biggest difference in chi1 in the 4 r groups in  transformation cycle.

    use keyword arguments 'transformation_1', 'transformation_2'

    Returns:
        float: biggest circle diff in chi1
    """
    property_key = kwargs["transformation_1"].compound_1.Properties.CHI2
    return _get_max_diagonal_delta_from_compounds(property_key, **kwargs)


def get_combinations(
    **kwargs: Transformation,
) -> list[tuple[Transformation, Transformation]]:
    """Get all for combinations of transformations to describe a compound.

    use keyword arguments transformation_1 ,2,3,4 for this to work.

    Returns:
        list[tuple[Transformation, Transformation]]: transformations
    """
    t1 = kwargs["transformation_1"]
    t2 = kwargs["transformation_2"]
    t3 = kwargs["transformation_3"]
    t4 = kwargs["transformation_4"]
    return [
        (t1, t2),
        (t1.get_reverse(), t3),
        (t2.get_reverse(), t4),
        (t3.get_reverse(), t4.get_reverse()),
    ]


def _substituents_in_same_ring_system(
    molecule: Molecule,
    index_1: int,
    index_2: int,
    aromatic_indices: set[int],
    fused_rings: list[set[int]],
) -> bool:
    ringinfo = molecule.GetRingInfo()
    if ringinfo.AreAtomsInSameRing(index_1, index_2) and all(
        i in aromatic_indices for i in (index_1, index_2)
    ):
        return True
    return any(all(v in ring for v in (index_1, index_2)) for ring in fused_rings)


def substituents_in_same_ring_system(**kwargs: Transformation) -> bool:
    """Check whether the two r groups in a cricle are connected to the same ring.

    Use keyword arguments transformation_1, 2, 3 and 4 for this to work.

    Returns:
        bool: True if both r groups are attached to the same ring.
    """
    for pot_transf_1, pot_transf_2 in get_combinations(**kwargs):
        anchor_1 = pot_transf_1.compound_1_anchor
        anchor_2 = pot_transf_2.compound_1_anchor
        compound = pot_transf_1.compound_1
        if not all(is_unique(a) for a in (anchor_1, anchor_2)):
            continue
        if not all(len(a) == 1 for a in (anchor_1, anchor_2)):
            continue
        return _substituents_in_same_ring_system(
            molecule=compound.rdkit_molecule,
            index_1=anchor_1[0],
            index_2=anchor_2[0],
            aromatic_indices=compound.get_property(
                compound.Properties.AROMATIC_INDICES,
            ),
            fused_rings=compound.get_property(compound.Properties.FUSED_RING_INDICES),
        )
    return False
