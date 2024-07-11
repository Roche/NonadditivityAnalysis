"""Utilitary functions for the classification workflow."""

from collections.abc import Sequence

from rdkit import Chem


def list_empty(nested_list: Sequence) -> bool:
    """Check whether a list or nested lists are empty.

    Args:
        nested_list (list): list to check.

    Returns:
        bool: True when all nested lists are empty
    """
    if isinstance(nested_list, list):
        return all(map(list_empty, nested_list))
    return False


def is_unique(anchor: int | Sequence[int | Sequence[int]]) -> int:
    """Check whether an anchor is unique.

    Args:
        anchor (int | Sequence[int | Sequence[int]]): list of atom indices.

    Returns:
        int: 1 if unique and length 1, 0 if unique and length > 1, -1 if not unique.
    """
    if not anchor or list_empty(anchor):  # type: ignore
        return -1
    assert isinstance(anchor, Sequence)
    if not isinstance(anchor[0], int):
        return -1
    return int(len(anchor) == 1)


def is_match_unique(
    match: int | Sequence[int | Sequence[int]],
) -> bool:
    """Return True if the list is not nested and not empty.

    Args:
        match (int | Sequence[int | Sequence[int]]): list to check.

    Returns:
        bool: True if list not nested and not empty.
    """
    if isinstance(match, int):
        return True
    if list_empty(match):
        return False
    return isinstance(match[0], int)


def convert_smarts_to_smiles(smarts: str) -> str:
    """Convert smarts to smiles.

    Crudely converts SMARTS to SMILES by creating a rdkit Molecule
    object from the SMARTS and writing it to SMILES.

    Args:
        smarts (str): input SMARTS

    Returns:
        str: SMILES
    """
    try:
        return Chem.MolToSmiles(  # type: ignore pylint: disable=E1101
            Chem.MolFromSmarts(smarts),  # type: ignore pylint: disable=E1101
        )
    except Exception as exc:
        raise TypeError from exc
