"""Implement Transformation class for classification.

Holds Class for Describing Properties on a Transformation Level.
This is used for the NonadditivityAnalysis Package.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

from nonadditivity.classification.ortho_classification import (
    ortho_substituent_exchanged,
    ortho_substituent_introduced,
)
from nonadditivity.classification.rgroup_distance import (
    get_constant_and_transformation_indices,
    get_r_group_anchor,
)
from nonadditivity.classification.transformation_classification import (
    calculate_fp_similarity,
    get_num_cuts,
    get_num_heavy_atoms_rgroups,
    is_h_replaced,
    stereo_classify_transformation,
    tertiary_amide_generated,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nonadditivity.classification.classification_classes.compound import Compound


class Transformation:
    """Class holding transformation classification.

    Transformation class holds two Compound object as well as strings
    for the constant part as well as the transformation part of the
    Transformation. It also calculates the properties enumerated in
    Properties.
    """

    class Properties(Enum):
        """Enumerates type of compound classifications."""

        IS_H_REPLACED = auto()
        NUM_HEAVY_ATOMS_IN_RGROUPS = auto()
        ORTHO_SUBSTITUENT_CHANGES = auto()
        ORTHO_SUBSTITUENT_INTRODUCED = auto()
        TERTIARY_AMIDE_FORMED = auto()
        TRANSFORMATION_STEREO = auto()
        MFP2_SIMILARITY = auto()
        NUM_MMPDB_CUTS = auto()

    classification_function: ClassVar[dict[Properties, Callable]] = {
        Properties.IS_H_REPLACED: is_h_replaced,
        Properties.NUM_HEAVY_ATOMS_IN_RGROUPS: get_num_heavy_atoms_rgroups,
        Properties.ORTHO_SUBSTITUENT_CHANGES: ortho_substituent_exchanged,
        Properties.ORTHO_SUBSTITUENT_INTRODUCED: ortho_substituent_introduced,
        Properties.TERTIARY_AMIDE_FORMED: tertiary_amide_generated,
        Properties.TRANSFORMATION_STEREO: stereo_classify_transformation,
        Properties.MFP2_SIMILARITY: calculate_fp_similarity,
        Properties.NUM_MMPDB_CUTS: get_num_cuts,
    }

    classification_keys: ClassVar[dict[Properties, str]] = {
        Properties.IS_H_REPLACED: "h_replaced",
        Properties.NUM_HEAVY_ATOMS_IN_RGROUPS: "num_heavy_atoms_rgroups",
        Properties.ORTHO_SUBSTITUENT_CHANGES: "ortho_substituent_exchanged",
        Properties.ORTHO_SUBSTITUENT_INTRODUCED: "ortho_substituent_introduced",
        Properties.TRANSFORMATION_STEREO: "transformation_stere_classification",
        Properties.TERTIARY_AMIDE_FORMED: "tertiary_amide_formed",
        Properties.MFP2_SIMILARITY: "mfp_2_similarity",
        Properties.NUM_MMPDB_CUTS: "num_mmpdb_cuts",
    }

    def __init__(
        self,
        compound_1: Compound,
        compound_2: Compound,
        constant_smarts: str,
        transformation_smarts: str,
    ) -> None:
        """Creat transformation instance.

        creates Transformation object holding 2 compounds
        as well as smarts for constant and transformation part.
        Upon instanciation, the indices for the constant and
        transformation part are calculatd for the rdkit molecules.

        Args:
            compound_1 (Compound): compound 1
            compound_2 (Compound): compound 2
            constant_smarts (str): constant smarts pattern
            transformation_smarts (str): transformation smarts pattern
        """
        self.compound_1 = compound_1
        self.compound_2 = compound_2
        self.constant_smarts = constant_smarts
        self.transformation_smarts = transformation_smarts.split(">>")

        (
            self.compound_1_constant_indices,
            self.compound_1_transformation_indices,
        ) = get_constant_and_transformation_indices(
            transformation_smarts=self.transformation_smarts[0],
            constant_smarts=constant_smarts,
            molecule=compound_1.rdkit_molecule,
        )

        (
            self.compound_2_constant_indices,
            self.compound_2_transformation_indices,
        ) = get_constant_and_transformation_indices(
            transformation_smarts=self.transformation_smarts[1],
            constant_smarts=constant_smarts,
            molecule=compound_2.rdkit_molecule,
        )

        self.compound_1_anchor = get_r_group_anchor(
            constant_indices=self.compound_1_constant_indices,
            transformation_indices=self.compound_1_transformation_indices,
        )
        self.compound_2_anchor = get_r_group_anchor(
            constant_indices=self.compound_2_constant_indices,
            transformation_indices=self.compound_2_transformation_indices,
        )
        self._classification: dict[
            Transformation.Properties,
            bool | int | list[int] | float,
        ] = {}

    def get_property(
        self,
        key: Properties,
    ) -> bool | int | list[int] | float:
        """Calculate property.

        Get a Property of a Transformation, all in Transformation.Properties are
        all-owed.

        Args:
            key (CompoundProperties): property name

        Returns:
            Union[bool,int]: Transformation porperty
        """
        try:
            return self._classification[key]
        except KeyError:
            self._classification[key] = Transformation.classification_function[key](
                compounds=[self.compound_1, self.compound_2],
                transformation_smarts=self.transformation_smarts,
                constant_smarts=self.constant_smarts,
                constant_indices=[
                    self.compound_1_constant_indices,
                    self.compound_2_constant_indices,
                ],
                anchors=(self.compound_1_anchor, self.compound_2_anchor),
            )
            return self.get_property(key=key)

    def classify(self) -> None:
        """Calculate all properties enumerated in Transformation.Properties."""
        for prop in Transformation.Properties:
            if prop not in self._classification:
                self.get_property(prop)

    def get_classification_dict(
        self,
        force_classification: bool = True,
    ) -> dict[Properties, bool | int | list[int] | float]:
        """Get dictionary with already calculated properties.

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
    ) -> dict[str, bool | int | list[int] | float]:
        """Get dictionary with strings for dataframe and calcualted properties.

        Return a dictionary with the properties already
        calculated for the compound with strings as keys
        instead of the Compound.Properties

        Args:
            force_classification (bool, optional): classifies the
            whole transformation. Defaults to True.

        Returns:
            dict[str, int]: calculated properties 'nicely' formatted.
        """
        if force_classification:
            self.classify()
        return {
            Transformation.classification_keys[key]: value
            for key, value in self._classification.items()
        }

    def get_reverse(self) -> Transformation:
        """Get reversed transformation object.

        Returns a Transformation object, that desctibes the
        transformation that is the reverse of this objects
        transforamtion.

        Returns:
            Transformation: reverse transformation.
        """
        transformation_smart = (
            f"{self.transformation_smarts[1]}>>{self.transformation_smarts[0]}"
        )
        return Transformation(
            compound_1=self.compound_2,
            compound_2=self.compound_1,
            constant_smarts=self.constant_smarts,
            transformation_smarts=transformation_smart,
        )
