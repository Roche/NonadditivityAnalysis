"""Test nonadditivity.classification.utils."""
import pytest

from nonadditivity.classification.utils import (
    convert_smarts_to_smiles,
    is_match_unique,
    is_unique,
    list_empty,
)


@pytest.mark.parametrize(
    "test_val, sol",
    [
        ([], True),
        ([[[]]], True),
        ([[], []], True),
        ([3], False),
        ([[], [3]], False),
        ("a", False),  # type: ignore
    ],
)
def test_list_empty(test_val: list | str, sol: int) -> None:
    """Test nonadditivity.classification.utils:list_empty.

    Args:
        test_val (list | str): test input
        sol (int): expected output.
    """
    assert list_empty(test_val) == sol


def test_convert_smarts_to_smiles() -> None:
    """Test nonadditivity.classification.utils:convert_smarts_to_smiles."""
    with pytest.raises(TypeError):
        convert_smarts_to_smiles(2)  # type: ignore
    assert convert_smarts_to_smiles("[#6]1=[#6]-[#6]=[#6]-[#6]=[#6]1") == "C1=CC=CC=C1"


@pytest.mark.parametrize(
    "test_val, sol",
    [
        ([], -1),
        ([[[]]], -1),
        ([[], []], -1),
        ([3], 1),
        ([[], [3]], -1),
        ([2, 3], 0),  # type: ignore
    ],
)
def test_is_unique(test_val: list, sol: int) -> None:
    """Test nonadditivity.classification.utils:is_unique.

    Args:
        test_val (list): test input
        sol (int): expected output
    """
    assert is_unique(test_val) == sol


@pytest.mark.parametrize(
    "test_val, sol",
    [
        (2, True),
        ([], False),
        ([3], True),
        ([3, 3], True),
        ([[3]], False),
    ],
)
def test_is_match_unique(test_val: int | list, sol: bool) -> None:
    """Test nonadditivity.classification.utils:is_match_unique.

    Args:
        test_val (int | list): test value
        sol (bool): expected output
    """
    assert is_match_unique(test_val) == sol
