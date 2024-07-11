"""Testing nonadditivity.workflows.mmpdb_helper."""

import filecmp
from pathlib import Path

import numpy as np
import pytest

from nonadditivity.utils.errors import FragmentationError, IndexingError
from nonadditivity.workflow.mmpdb_helper import (
    read_raw_mmps,
    run_mmpdb_fragment,
    run_mmpdb_index,
    run_mmpdlib_code,
)
from tests._utils import assert_exists_and_remove


@pytest.mark.slow()
@pytest.mark.parametrize(
    "cache, frag, struct, error",
    [
        (False, "temp_frag", "test_smiles", None),
        (True, "test_frag", "test_smiles", None),
        (True, "temp_frag", "test_smiles", FragmentationError),
        (True, "test_faultyfrag", "test_smiles", FragmentationError),
        (True, "test_faultyfrag2", "test_smiles", FragmentationError),
        (False, "temp_frag", "test_faultysmiles", FragmentationError),
    ],
)
def test_run_mmpdb_fragment(
    cache: bool,
    frag: str,
    struct: str,
    error: Exception | None,
    paths: dict[str, Path],
) -> None:
    """Test nonadditivity.workflows.mmpdb_helper:run_mmpdb_fragment.

    Args:
        cache (bool): cache input flat
        frag (str): fragment test file
        struct (str): structure test file
        error (Exception | None): expected error
        paths (dict[str, Path]): dict mapping string to test paths.
    """
    if error is not None:
        with pytest.raises(error):
            run_mmpdb_fragment(
                cache=cache,
                fragments_file=paths[frag],
                structure_file=paths[struct],
            )
        return
    run_mmpdb_fragment(
        cache=cache,
        fragments_file=paths[frag],
        structure_file=paths[struct],
    )
    if frag == "temp_frag":
        assert_exists_and_remove(paths["temp_frag"])


@pytest.mark.slow()
@pytest.mark.parametrize(
    "frag, error",
    [("test_frag", None), ("test_faultyfrag", IndexingError)],
)
def test_index_mmps(frag: str, error: Exception | None, paths: dict[str, Path]) -> None:
    """Test nonadditivity.workflows.mmpdb_helper:run_mmpdb_index.

    Args:
        frag (str): test fragmentfile ipnut
        error (Exception | None): expected error
        paths (dict[str, Path]): dict mapping string to test paths
    """
    if error is not None:
        with pytest.raises(error):
            run_mmpdb_index(
                fragments_file=paths[frag],
                mmp_outputfile=paths["temp_mmp"],
                max_variable_heavies=10,
            )
        return
    run_mmpdb_index(
        fragments_file=paths[frag],
        mmp_outputfile=paths["temp_mmp"],
        max_variable_heavies=10,
    )
    assert_exists_and_remove(paths["temp_mmp"])


def testread_raw_mmps(paths: dict[str, Path]) -> None:
    """Test nonadditivity.workflows.mmpdb_helper:read_raw_mmps.

    Args:
        paths (dict[str, Path]): dict str -> pahts for test files
    """
    mmps = read_raw_mmps(paths["test_mmp"])
    assert np.array_equal(
        mmps.columns,
        [
            "SMILES_LHS",
            "SMILES_RHS",
            "ID_LHS",
            "ID_RHS",
            "TRANSFORMATION",
            "CONSTANT",
        ],
    )


@pytest.mark.slow()
@pytest.mark.parametrize("cache, frag", [(True, "test_frag"), (False, "temp_frag")])
def test_run_mmpdlib_code(cache: bool, frag: str, paths: dict[str, Path]) -> None:
    """Test nonadditivity.workflows.mmpdb_helper:run_mmpdlib_code.

    Args:
        cache (bool): cache test input value
        frag (str): fragment file test input value
        paths (dict[str, Path]): dict mapping str to test file paths.
    """
    run_mmpdlib_code(
        cache=cache,
        fragments_file=paths[frag],
        mmp_outputfile=paths["temp_mmp"],
        max_variable_heavies=16,
        structure_file=paths["test_smiles"],
    )
    assert filecmp.cmp(paths["temp_mmp"], paths["test_mmp"])
    assert_exists_and_remove(path=paths["temp_mmp"])
    if frag == "temp_frag":
        assert filecmp.cmp(paths["temp_frag"], paths["test_frag"])
        assert_exists_and_remove(path=paths["temp_frag"])
