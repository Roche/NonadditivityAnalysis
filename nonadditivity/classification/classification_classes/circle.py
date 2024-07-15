"""Implement Circle class for classification.

Holds Class for Describing Properties on a circle Level.
This is used for the NonadditivityAnalysis Package.
"""

from collections.abc import Callable
from enum import Enum, auto
from typing import ClassVar

from nonadditivity.classification.circle_classification import (
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
    get_sp3_carbon_diff,
    get_tertiary_amide_formed,
    get_tpsa_diff,
    substituents_in_same_ring_system,
)
from nonadditivity.classification.classification_classes.compound import Compound
from nonadditivity.classification.classification_classes.transfromation import (
    Transformation,
)
from nonadditivity.classification.ortho_classification import (
    get_ortho_transformation_changing,
    get_ortho_transformation_introduced,
)
from nonadditivity.classification.rgroup_distance import get_num_atoms_between_rgroups


class Circle:
    """Object for classifying circles.

    The Circle Class holds all relevant information for classifying a NAA double
    transformation cycle. It holds the four transformation objects of a circle
    and hence also the four compounds the circle is made from.
    """

    class Properties(Enum):
        """Enumerates type of compound classifications."""

        DISTANCE_BETWEEN_R_GROUPS = auto()
        H_BOND_DONORS_DIFF = auto()
        H_BOND_ACCEPTORS_DIFF = auto()
        FORMAL_CHARGE_DIFF = auto()
        POLAR_SURFACE_AREA_DIFF = auto()
        NUM_ROTATABLE_BONDS_DIFF = auto()
        SP3_CARBON_DIFF = auto()
        LOG_P_DIFF = auto()
        CHI0_DIFF = auto()
        CHI1_DIFF = auto()
        CHI2_DIFF = auto()
        NUM_HEAVY_ATOMS_DIFF = auto()
        HAS_TRANSFORMATION_AT_ORTHO = auto()
        HAS_ORTHO_SUBSTITUENT_INTRODUCED = auto()
        HAS_TERTIARY_AMIDE_FORMED = auto()
        HAS_INVERSION_IN_TRANSFORMATION = auto()
        MAX_NUM_MMPDB_CUTS = auto()
        MAX_HEAVY_ATOM_IN_TRANSFORMATION = auto()
        COMPOUND_STEREO_CLASSIFICATION = auto()
        MIN_TANIMOTO = auto()
        SUBSTITUENT_ON_SAME_RING_SYSYTEM = auto()

    classification_function: ClassVar[dict[Properties, Callable]] = {
        Properties.DISTANCE_BETWEEN_R_GROUPS: get_num_atoms_between_rgroups,
        Properties.H_BOND_DONORS_DIFF: get_hbond_donor_diff,
        Properties.H_BOND_ACCEPTORS_DIFF: get_hbond_acceptor_diff,
        Properties.FORMAL_CHARGE_DIFF: get_formal_charge_diff,
        Properties.POLAR_SURFACE_AREA_DIFF: get_tpsa_diff,
        Properties.NUM_ROTATABLE_BONDS_DIFF: get_num_rot_bonds_diff,
        Properties.SP3_CARBON_DIFF: get_sp3_carbon_diff,
        Properties.LOG_P_DIFF: get_log_p_diff,
        Properties.NUM_HEAVY_ATOMS_DIFF: get_num_heavy_atoms_diff,
        Properties.CHI0_DIFF: get_chi0_diff,
        Properties.CHI1_DIFF: get_chi1_diff,
        Properties.CHI2_DIFF: get_chi2_diff,
        Properties.HAS_TRANSFORMATION_AT_ORTHO: get_ortho_transformation_changing,
        Properties.HAS_ORTHO_SUBSTITUENT_INTRODUCED: get_ortho_transformation_introduced,
        Properties.HAS_TERTIARY_AMIDE_FORMED: get_tertiary_amide_formed,
        Properties.HAS_INVERSION_IN_TRANSFORMATION: circle_has_stereoinversion_in_transformation,
        Properties.MAX_NUM_MMPDB_CUTS: get_max_num_mmpdb_cuts,
        Properties.MAX_HEAVY_ATOM_IN_TRANSFORMATION: get_max_num_heavy_atom_in_transformation,
        Properties.COMPOUND_STEREO_CLASSIFICATION: get_compound_stereocenter_classification,
        Properties.MIN_TANIMOTO: get_min_transformation_tanimoto,
        Properties.SUBSTITUENT_ON_SAME_RING_SYSYTEM: substituents_in_same_ring_system,
    }
    classification_keys: ClassVar[dict[Properties, str]] = {
        Properties.DISTANCE_BETWEEN_R_GROUPS: "num_atoms_between_r_groups",
        Properties.H_BOND_DONORS_DIFF: "hbond_donor_diff",
        Properties.H_BOND_ACCEPTORS_DIFF: "hbond_acceptor_diff",
        Properties.FORMAL_CHARGE_DIFF: "formal_charge_diff",
        Properties.POLAR_SURFACE_AREA_DIFF: "tpsa_diff",
        Properties.NUM_ROTATABLE_BONDS_DIFF: "num_rot_bonds_diff",
        Properties.SP3_CARBON_DIFF: "sp3_carbon_diff",
        Properties.LOG_P_DIFF: "log_p_diff",
        Properties.CHI0_DIFF: "chi0_diff",
        Properties.CHI1_DIFF: "chi1_diff",
        Properties.CHI2_DIFF: "chi2_diff",
        Properties.NUM_HEAVY_ATOMS_DIFF: "num_heavy_atoms_diff",
        Properties.HAS_TRANSFORMATION_AT_ORTHO: "transformation_at_ortho",
        Properties.HAS_ORTHO_SUBSTITUENT_INTRODUCED: "ortho_substituent_introduced",
        Properties.HAS_TERTIARY_AMIDE_FORMED: "tertiary_amide_formed",
        Properties.HAS_INVERSION_IN_TRANSFORMATION: "has_stereoinversion_in_transformation",
        Properties.MAX_NUM_MMPDB_CUTS: "max_num_mmpdb_cuts",
        Properties.MAX_HEAVY_ATOM_IN_TRANSFORMATION: "max_num_heavy_atom_in_transformation",
        Properties.COMPOUND_STEREO_CLASSIFICATION: "compound_stereocenter_classification",
        Properties.MIN_TANIMOTO: "min_transformation_tanimoto",
        Properties.SUBSTITUENT_ON_SAME_RING_SYSYTEM: "substituents_in_same_ring_system",
    }

    def __init__(
        self,
        transformation_1: Transformation,
        transformation_2: Transformation,
        transformation_3: Transformation,
        transformation_4: Transformation,
    ) -> None:
        """Instanciate circle classification object.

        Args:
            transformation_1 (Transformation): transformation 1 in the circle
            transformation_2 (Transformation): transformation 2 in the circle
            transformation_3 (Transformation): transformation 3 in the circle
            transformation_4 (Transformation): transformation 4 in the circle
        """
        self.transformation_1 = transformation_1
        self.transformation_2 = transformation_2
        self.transformation_3 = transformation_3
        self.transformation_4 = transformation_4
        self._classification: dict[Circle.Properties, int] = {}

    def compounds(self) -> list[Compound]:
        """Get compounds in the circle.

        Returns list of compounds as they are named in the get_circles function in
        nonadditivty.nonadditivty_core

        Returns:
            list[Compound]: list of compounds in circle
        """
        return [
            self.transformation_1.compound_1,
            self.transformation_1.compound_2,
            self.transformation_4.compound_2,
            self.transformation_4.compound_1,
        ]

    def get_property(self, key: Properties) -> int | tuple[int] | bool:
        """Calculate a Property of a compound, all in Circle.Properties are allowed.

        Args:
            key (CompoundProperties): property name

        Returns:
            int: compound porperty
        """
        try:
            return self._classification[key]
        except KeyError:
            self._classification[key] = Circle.classification_function[key](
                transformation_1=self.transformation_1,
                transformation_2=self.transformation_2,
                transformation_3=self.transformation_3,
                transformation_4=self.transformation_4,
            )
            return self.get_property(key=key)

    def classify(self) -> None:
        """Calculate all properties enumerated in Circle.Properties."""
        for prop in Circle.Properties:
            if prop not in self._classification:
                _ = self.get_property(prop)

    def get_classification_dict(
        self,
        force_classification: bool = True,
    ) -> dict[Properties, int]:
        """Get dictionary with the properties already calculated for the Circle.

        Args:
            force_classification (bool, optional): Forces calculation of all properties.
            Defaults to True.

        Returns:
            dict[str, int]: calculated properties
        """
        if force_classification:
            self.classify()
        return self._classification

    def get_formatted_classification_dict(
        self,
        force_classification: bool = True,
    ) -> dict[str, int]:
        """Get dictionary with strings for dataframe and calcualted properties.

        Return a dictionary with the properties already calculated for the compound
        with strings as keys instead of the Compound.Properties

        Args:
            force_classification (bool, optional): whether to force all possible
            classification. Defaults to True.

        Returns:
            dict[str, int]: formatted classificaton dictionary.
        """
        if force_classification:
            self.classify()
        return {
            Circle.classification_keys[key]: value
            for key, value in self._classification.items()
        }
