"""Implement Compound class for classification.

Holds Class for Describing Properties on a compound Level.
This is used for the NonadditivityAnalysis Package.
"""

from collections.abc import Callable
from enum import Enum, auto
from typing import ClassVar

from rdkit import Chem

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
from nonadditivity.classification.ortho_classification import (
    get_num_ortho_configurations,
    get_ortho_indices,
)
from nonadditivity.utils.types import Molecule


class Compound:
    """Compound class holds an compound_Id, smiles and a rdkit molecule.

    It also calculates the properties enumerated in Properties
    """

    class Properties(Enum):
        """Enumerates type of compound classifications."""

        NUM_HEAVY_ATOMS = auto()
        NUM_ORTHO_CONFIGURATIONS = auto()
        HAS_UNASSIGNED_STEREOCENTERS = auto()
        NUM_STERO_CENTERS = auto()
        ORTHO_INDICES = auto()
        AROMATIC_INDICES = auto()
        FUSED_RING_INDICES = auto()
        CHARGE = auto()
        NUM_HBA = auto()
        NUM_HBD = auto()
        TPSA = auto()
        LOGP = auto()
        CHI0 = auto()
        CHI1 = auto()
        CHI2 = auto()
        NUM_ROT_BONDS = auto()
        NUM_SP3_CARBONS = auto()
        MORGANFP = auto()

    classification_function: ClassVar[dict[Properties, Callable]] = {
        Properties.NUM_HEAVY_ATOMS: get_num_heavy_atoms,
        Properties.NUM_STERO_CENTERS: get_num_stereo_centers,
        Properties.NUM_ORTHO_CONFIGURATIONS: get_num_ortho_configurations,
        Properties.ORTHO_INDICES: get_ortho_indices,
        Properties.AROMATIC_INDICES: get_aromatic_indices,
        Properties.FUSED_RING_INDICES: get_fused_ring_indices,
        Properties.HAS_UNASSIGNED_STEREOCENTERS: has_unassigned_stereocenters,
        Properties.CHARGE: get_charge,
        Properties.NUM_HBA: get_num_h_bond_acceptors,
        Properties.NUM_HBD: get_num_h_bond_donors,
        Properties.TPSA: get_polar_surface_area,
        Properties.LOGP: get_log_p,
        Properties.CHI0: get_chi0,
        Properties.CHI1: get_chi1,
        Properties.CHI2: get_chi2,
        Properties.NUM_ROT_BONDS: get_num_rotatable_bonds,
        Properties.NUM_SP3_CARBONS: get_num_sp3_carbons,
        Properties.MORGANFP: get_morgan_fp,
    }

    classification_keys: ClassVar[dict[Properties, str]] = {
        Properties.NUM_HEAVY_ATOMS: "num_heavy_atoms",
        Properties.NUM_STERO_CENTERS: "num_stereo_centers",
        Properties.NUM_ORTHO_CONFIGURATIONS: "num_ortho_configurations",
        Properties.ORTHO_INDICES: "ortho_indices",
        Properties.AROMATIC_INDICES: "aromatic_indices",
        Properties.FUSED_RING_INDICES: "fused_ring_indices",
        Properties.HAS_UNASSIGNED_STEREOCENTERS: "has_unassigned_stereocenters",
        Properties.CHARGE: "charge",
        Properties.NUM_HBA: "num_hba",
        Properties.NUM_HBD: "num_hdb",
        Properties.TPSA: "tpsa",
        Properties.LOGP: "log_p",
        Properties.CHI0: "chi0",
        Properties.CHI1: "chi1",
        Properties.CHI2: "chi2",
        Properties.NUM_ROT_BONDS: "num_rotatable_bonds",
        Properties.NUM_SP3_CARBONS: "num_sp3_carbons",
        Properties.MORGANFP: "morgan_fp",
    }

    def __init__(
        self,
        molecule: Molecule,
        compound_id: str,
        smiles: str,
    ) -> None:
        """Instanciate compound object holding an id and an rdkit molecule and smiles.

        Args:
            molecule (Molecule): rdkit molecule
            compound_id (str): compound id
            smiles (str): compound smiles
        """
        self.rdkit_molecule = Chem.MolFromSmiles(  # type: ignore pylint: disable=E1101
            Chem.MolToSmiles(molecule),  # type: ignore pylint: disable=E1101
        )
        self.compound_id = compound_id
        self.smiles = smiles
        self._classification: dict[
            Compound.Properties,
            int | list[tuple[set[int], ...]],
        ] = {}

    def get_property(
        self,
        key: Properties,
    ) -> int | list[tuple[set[int], ...]] | set[int] | list[set[int]]:
        """Get a Property of a compound, all in Compound.Properties are allowed.

        Args:
            key (CompoundProperties): property name

        Returns:
            int: compound porperty
        """
        try:
            return self._classification[key]
        except KeyError:
            self._classification[key] = Compound.classification_function[key](
                molecule=self.rdkit_molecule,
            )
            return self.get_property(key=key)

    def classify(self) -> None:
        """Calculate all properties enumerated in.Compound.Properties."""
        for prop in Compound.Properties:
            if prop not in self._classification:
                self.get_property(prop)

    def get_classification_dict(
        self,
        force_classification: bool = True,
    ) -> dict[Properties, int | list[tuple[set[int], ...]]]:
        """Get dictionary with the properties already calculated for the compound.

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
    ) -> dict[str, int | list[tuple[set[int], ...]]]:
        """Return dict with key for datframe and calculated properties.

        Return a dictionary with the properties already calculated for the compound
        with strings as keys instead of the Compound.Properties

        Args:
            force_classification (bool, optional): whether to force all classification.
            Defaults to True.

        Returns:
            dict[str, int]: formatted classification dict
        """
        if force_classification:
            self.classify()
        return {
            Compound.classification_keys[key]: value
            for key, value in self._classification.items()
        }
