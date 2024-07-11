"""Test nonadditvity.nonadditivity_workflow."""

from pathlib import Path

import pytest

from nonadditivity.nonadditivity_workflow import run_nonadd_calculation
from nonadditivity.utils.commandline import InputOptions
from tests._utils import assert_exists_and_remove, files_equal, same_size


@pytest.mark.slow()
@pytest.mark.parametrize(
    "inopt, solfiles",
    [
        ("input_options", ("test_naa", "test_c2c", "test_per_cpd")),
        (
            "input_options_classify",
            ("test_naa_classify", "test_c2c_classify", "test_per_cpd_classify"),
        ),
        (
            "input_options_update",
            ("test_naa_series", "test_c2c", "test_per_cpd_series"),
        ),
        (
            "input_options_multiprops",
            ("test_naa_mult", "test_c2c_mult", "test_per_cpd_mult"),
        ),
        (
            "input_options_multiprops_censored",
            ("test_naa_mult_wc", "test_c2c_mult_wc", "test_per_cpd_mult_wc"),
        ),
    ],
)
def test_run_nonadd_calculation(
    inopt: str,
    solfiles: tuple[str, str, str],
    request: pytest.FixtureRequest,
    input_options_update: InputOptions,
    caplog: pytest.LogCaptureFixture,
    paths: dict[str, Path],
) -> None:
    """Test nonadditvity.nonadditivity_workflow:run_nonadd_calculation.

    Args:
        inopt (str): input options
        solfiles (tuple[str, str, str]): dict keys for solution files
        request (pytest.FixtureRequest): pytest magic
        input_options_update (InputOptions): for checking
        caplog (pytest.LogCaptureFixture): pytest magic
        paths (dict[str, Path]): paths to test files
    """
    new_files = ["NAA_output.csv", "c2c.csv", "perCompound.csv"]
    input_params = request.getfixturevalue(inopt)
    run_nonadd_calculation(input_options=input_params)
    if input_params == input_options_update:
        assert (
            "Was not able to locate results from previous fragmentation." in caplog.text
        )
        assert "Will redo all fragmentation." in caplog.text

    check_files = [
        ("fragments.json", paths["test_frag"]),
        ("ligands.smi", paths["test_smiles"]),
        ("mmp_raw.csv", paths["test_mmp"]),
    ]
    for new_file, shouldbefile in check_files:
        assert files_equal(path1=paths["dir"] / new_file, path2=shouldbefile)
        assert_exists_and_remove(path=paths["dir"] / new_file)

    for new_file, sol in zip(new_files, (solfiles[0], solfiles[1], solfiles[2])):
        assert same_size(
            path1=paths["dir"] / new_file,
            path2=paths[sol],
            rel=0.01,
        )
        assert_exists_and_remove(path=paths["dir"] / new_file)
    assert_exists_and_remove(
        path=paths["dir"] / "problem_smiles.smi",
        assert_not_empty=False,
    )
    assert_exists_and_remove(paths["temp_canonical_transf"])
    assert_exists_and_remove(paths["temp_canonical_naa"])
