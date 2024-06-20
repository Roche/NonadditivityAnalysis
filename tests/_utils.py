"""Some convenience functions for the tests of the nonadditivity package."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from nonadditivity.utils.types import Molecule


def assert_exists_and_remove(
    path: Path | str,
    assert_not_empty: bool = True,
) -> None:
    """Assert that file exists, remove file and assert deletion.

    Args:
        path (Path | str): file to be checked.
        assert_not_empty (bool): assert that the file is not empty. Defaults to True.

    Raises:
        AssertionError: If path is not existing in the beginning,
            or the file is still existing after removing.
    """
    assert os.path.exists(path)
    if assert_not_empty:
        assert os.stat(path).st_size != 0
    os.remove(path)
    assert not os.path.exists(path)


def same_size(
    path1: Path | str,
    path2: Path | str,
    rel: float = 0.1,
) -> bool:
    """Assert that two files have the same size.

    Args:
        path1 (Path | str): path to file 1
        path2 (Path | str): path to file 2
        rel (float): relative allowed difference in file size. Defaults to 0.1

    Raises:
        AssertionError: if files are not the same size
    """
    return os.stat(path1).st_size == pytest.approx(
        os.stat(path2).st_size,
        rel=rel,
    )


def files_equal(path1: Path | str, path2: Path | str) -> bool:
    """Check two files line by line and returns true when all lines are equal.

    Args:
        path1 (Path | str): File path 1
        path2 (Path | str): File path 2

    Returns:
        bool: wheter both files are the same
    """
    with open(path1, encoding="utf-8") as file_1, open(
        path2,
        encoding="utf=-8",
    ) as file_2:
        lines_f1, lines_f2 = file_1.readlines(), file_2.readlines()
        for line_1, line_2 in zip(lines_f1, lines_f2):
            if line_1 != line_2:
                return False
    return True


def raises_exceptions(
    test_object: Molecule | list[str],
    function: Callable,
    exception: Exception,
    **correct_keyword_wrong_type_argument,
) -> None:
    """Test that assertions are thrown.

    type error thronw when function is called with test_object.
    key error thrwon when function is called with false_keyword.
    exception thrown when function is called with correct keyword
    but wrong argument type.

    Args:
        test_object (Molecule | list[str]): object to test on
        function (Callable): function to call object on
        exception (Exception): exception raised by correct_keyword_wrong_type_argument
        **correct_keyword_wrong_type_argument: give a correct keyword
        but wrong argument type/
    """
    with pytest.raises(TypeError):
        function(test_object)  # type: ignore pylint: disable=E1121
    with pytest.raises(KeyError):
        function(false_keyword=test_object)
    with pytest.raises(exception):
        function(**correct_keyword_wrong_type_argument)
