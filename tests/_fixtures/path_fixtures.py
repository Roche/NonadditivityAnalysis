from pathlib import Path

import pytest


@pytest.fixture()
def paths() -> dict[str, Path]:
    directory = Path("tests/_test_files/")
    dictionary = {
        "dir": directory,
        # actual input files, named with test_*
        "test_frag": directory / "test_fragments.json",
        "test_faultyfrag": directory / "test_fragments_faulty.json",
        "test_faultyfrag2": directory / "test_fragments_faulty2.json",
        "test_smiles": directory / "test_smiles.smi",
        "test_faultysmiles": directory / "test_smiles_faulty.smi",
        "test_mmp": directory / "test_raw_mmp.csv",
        "test_input": directory / "test_pchembl_input.txt",
        "test_c2c": directory / "test_c2c.csv",
        "test_c2c_classify": directory / "test_c2c_classify.csv",
        "test_c2c_series": directory / "test_c2c_series.csv",
        "test_c2c_mult": directory / "test_c2c_mult.csv",
        "test_c2c_mult_wc": directory / "test_c2c_mult_wc.csv",
        "test_naa": directory / "test_naa.csv",
        "test_naa_classify": directory / "test_naa_classify.csv",
        "test_naa_series": directory / "test_naa_series.csv",
        "test_naa_mult": directory / "test_naa_mult.csv",
        "test_naa_mult_wc": directory / "test_naa_mult_wc.csv",
        "test_per_cpd": directory / "test_per_cpd.csv",
        "test_per_cpd_classify": directory / "test_per_cpd_classify.csv",
        "test_per_cpd_series": directory / "test_per_cpd_series.csv",
        "test_per_cpd_mult": directory / "test_per_cpd_mult.csv",
        "test_per_cpd_mult_wc": directory / "test_per_cpd_mult_wc.csv",
        # files that are created and subsequently deleted by tests
        # named with temp_*
        "temp_frag": directory / "temp_fragments.json",
        "temp_smiles": directory / "temp_smiles.smi",
        "temp_mmp": directory / "temp_mmp.csv",
        "temp_c2c": directory / "temp_c2c.csv",
        "temp_naa": directory / "temp_naa.csv",
        "temp_per_cpd": directory / "temp_per_cpd.csv",
        "temp_log_file": directory / "nonadditivity.log",
        "temp_canonical_transf": directory / "canonical_transformations.csv",
        "temp_canonical_naa": directory / "canonical_na_output.csv",
    }
    return {k: v.resolve() for k, v in dictionary.items()}
