"""Type aliases for rdkit Molecules and Atoms."""

from typing import TypeAlias

from rdkit.Chem import rdchem

Molecule: TypeAlias = rdchem.Mol  # pylint:disable=I1101
Atom: TypeAlias = rdchem.Atom  # pylint:disable=I1101
