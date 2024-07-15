"""Test nonadditvity.cli."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from nonadditivity.cli import main
from tests._utils import assert_exists_and_remove, files_equal, same_size


@pytest.mark.slow()
def test_main(cli_input_arguments: list[str], paths: dict[str, Path]) -> None:
    """Test nonadditvity.cli:main.

    Args:
        cli_input_arguments (list[str]): command line arguments
        paths (dict[str, Path]): path to solution files.
    """
    runner = CliRunner()
    result = runner.invoke(main, cli_input_arguments)
    assert result.exit_code == 0
    check_files = [
        (paths["dir"] / "fragments.json", paths["test_frag"]),
        (paths["dir"] / "ligands.smi", paths["test_smiles"]),
        (paths["dir"] / "mmp_raw.csv", paths["test_mmp"]),
    ]
    for new_file, shouldbefile in check_files:
        assert files_equal(new_file, shouldbefile)
        assert_exists_and_remove(new_file)
    check_files = [
        (paths["dir"] / "NAA_output.csv", paths["test_naa"]),
        (paths["dir"] / "perCompound.csv", paths["test_per_cpd_series"]),
        (paths["dir"] / "c2c.csv", paths["test_c2c"]),
    ]
    for new_file, shouldbefile in check_files:
        assert same_size(new_file, shouldbefile, rel=0.05)
        assert_exists_and_remove(new_file)
    assert_exists_and_remove(
        paths["dir"] / "problem_smiles.smi",
        assert_not_empty=False,
    )
