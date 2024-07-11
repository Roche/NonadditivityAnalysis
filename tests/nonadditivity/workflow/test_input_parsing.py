"""Test nonadditivity.workflows.input_parsing."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._typing import ArrayLike
from rdkit import RDConfig
from rdkit.Chem import SaltRemover

from nonadditivity.workflow.input_parsing import (
    _check_and_convert_activities,
    _check_for_censored_values,
    _get_salt_remover,
    _parse_input_smiles,
    _read_in_data,
    _remove_duplicate_structures,
    _remove_too_big_molecules,
    _remove_unknown_and_faulty_structures,
    parse_input_file,
)
from tests._utils import assert_exists_and_remove


@pytest.fixture()
def input_truths(paths: dict[str, Path]) -> dict[str, ArrayLike]:
    """Create dict from csv file.

    Args:
        paths (dict[str, Path]): paths to test files

    Returns:
        dict[str, ArrayLike]: columnnname -> values.
    """
    dataframe = pd.read_table(paths["test_input"])
    return {
        v: dataframe[v].to_numpy()
        for v in ("CMPD_TEST_ID", "TEST_PCHEMBL_VALUE", "TEST_SMILES", "Series")
    }


def test_get_salt_remover() -> None:
    """Test nonadditivity.workflows.input_parsing:get_salt_remover."""
    assert dir(_get_salt_remover()) == dir(
        SaltRemover.SaltRemover(
            defnFilename=os.path.join(RDConfig.RDDataDir, "Salts.txt"),
        ),
    )


def test_parse_input_smiles(smiles_dataframe: pd.DataFrame) -> None:
    """Test nonadditivity.workflows.input_parsing:parse_input_smiles.

    Args:
        smiles_dataframe (pd.DataFrame): dataframe containing smiles.
    """
    unknown, faulty, smiles_dataframe = _parse_input_smiles(smiles_dataframe)
    assert np.array_equal(
        smiles_dataframe.Num_Heavy_Atoms.to_numpy(),
        [6.0, 6.0, -1, -1, -1],
    )
    assert np.array_equal(
        smiles_dataframe.Molecular_Weight.to_numpy(),  # type: ignore
        np.array([78.11399999999999, 78.11399999999999, -1.0, -1.0, -1.0]),
    )
    assert np.array_equal(
        smiles_dataframe.SMILES.to_numpy(),  # type: ignore
        [
            "c1ccccc1",
            "c1ccccc1",
            "CCC.c1ccccc1",
            "c1ccccc1.Ag",
            "c1cccc1.Cl",
        ],
    )
    assert unknown == ["ID3"]
    assert np.array_equal(faulty, ["ID4", "ID5"])


@pytest.mark.parametrize(
    "inp, output",
    [(True, "Parsing Input SMILES"), (False, "")],
)
def test_logging(
    smiles_dataframe: pd.DataFrame,
    capfd: pytest.CaptureFixture[str],
    inp: bool,
    output: str,
) -> None:
    """Test logging in Test nonadditivity.workflows.input_parsing.

    Args:
        smiles_dataframe (pd.DataFrame): smiles data
        capfd (pytest.CaptureFixture[str]): captured log
        inp (bool): verbosity
        output (str): expected output.
    """
    _ = _parse_input_smiles(smiles_dataframe, verbose=inp)
    _, err = capfd.readouterr()
    assert output in err


@pytest.mark.parametrize(
    "column, values, solution, unit, colsol",
    [
        ("test_pchembl", [5, 4, 6], [5.0, 4.0, 6.0], None, ""),
        ("test_pIC50", [2, 1, 3], [11, 10, 12], "nm", ""),
        (
            "test_pm",
            [8, 7, 9],
            [-np.log10(8 * 1e-12), -np.log10(7 * 1e-12), -np.log10(9 * 1e-12)],
            "pm",
            "p",
        ),
        ("test_no_unit", [5, 4, 6], [5, 4, 6], "noconv", ""),
        ("negative_or_zero", [0.0, -1, 3], [-np.log10(3 * 1e-6)], "um", "p"),
    ],
)
def test_check_and_convert_activities(
    column: str,
    values: list[float],
    solution: list[float],
    unit: str | None,
    colsol: str,
) -> None:
    """Test nonadditivity.workflows.input_parsing:check_and_convert_activities.

    Args:
        column (str): column name
        values (list[float]): values to be converted
        solution (list[float]): expected solution
        unit (str | None): unit the values are in
        colsol (str): whether to add 'p' to the expected colname
    """
    data = pd.DataFrame()
    data[column] = values
    print(data)
    colnames, data = _check_and_convert_activities(
        dataframe=data,
        property_columns=[column],
        units=[unit],
    )
    print(data)
    assert np.array_equal(colnames, [f"{colsol}{column}"])
    assert np.array_equal(data[colnames[0]].to_numpy(), solution)


def test_remove_duplicate_structures(
    smiles_dataframe: pd.DataFrame,
    duplicate_dataframe: pd.DataFrame,
) -> None:
    """Test nonadditivity.workflows.input_parsing:remove_duplicate_structures.

    Args:
        smiles_dataframe (pd.DataFrame): dataframe no duplicates
        duplicate_dataframe (pd.DataFrame): datafrmae with duplicates
    """
    with pytest.raises(ValueError):
        duplicate_dataframe = _remove_duplicate_structures(duplicate_dataframe)
    smiles_dataframe = _remove_duplicate_structures(smiles_dataframe)
    assert np.array_equal(
        smiles_dataframe.index.to_numpy(),
        ["ID1", "ID3", "ID4", "ID5"],
    )


def test_remove_unknown_and_faulty_structures(
    smiles_dataframe: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Test nonadditivity.workflows.input_parsing:remove_unknown_and_faulty_structures.

    Args:
        smiles_dataframe (pd.DataFrame): dataframe np duplicate
        paths (dict[str, Path]): paths to test files.
    """
    smiles_dataframe = _remove_unknown_and_faulty_structures(
        smiles_dataframe,
        ["ID1", "ID2"],
        ["ID2", "ID3"],
        paths["temp_smiles"],
    )
    assert len(smiles_dataframe) == 2
    assert_exists_and_remove(str(paths["temp_smiles"]))


def test_read_in_data(
    input_truths: dict[str, ArrayLike],
    paths: dict[str, Path],  # pylint:disable=W0621
) -> None:
    """Test nonadditivity.workflows.input_parsing:read_in_data.

    Args:
        input_truths (dict[str, ArrayLike]): solution values
        paths (dict[str, Path]): paths to testfiles.
    """
    dataframe = _read_in_data(
        paths["test_input"],
        property_columns=["TEST_PCHEMBL_VALUE"],
    )
    assert np.array_equal(dataframe.SMILES.to_numpy(), input_truths["TEST_SMILES"])  # type: ignore
    assert np.array_equal(dataframe.Series.to_numpy(), input_truths["Series"])  # type: ignore
    assert np.array_equal(dataframe.index.to_numpy(), input_truths["CMPD_TEST_ID"])  # type: ignore
    assert np.array_equal(
        dataframe.TEST_PCHEMBL_VALUE.to_numpy(),  # type: ignore
        input_truths["TEST_PCHEMBL_VALUE"],  # type: ignore
    )

    dataframe = _read_in_data(
        paths["test_input"],
        property_columns=["TEST_PCHEMBL_VALUE"],
        series_column="Series",
    )
    assert np.array_equal(dataframe.Series.to_numpy(), input_truths["Series"])  # type: ignore


def test_check_for_censored_values(censored_dataframe: pd.DataFrame) -> None:
    """Test nonadditivity.workflows.input_parsing:check_for_censored_values.

    Args:
        censored_dataframe (pd.DataFrame): data with censored values.
    """
    censored_dataframe = _check_for_censored_values(censored_dataframe, ["VALUES"])
    assert np.array_equal(
        censored_dataframe.VALUES.to_numpy(),
        ["1", "2", "3", "4", "5", "5", "0", "0"],
    )
    assert np.array_equal(
        censored_dataframe.VALUES_Censors.to_numpy(),
        ["", "<", ">", "*", "*", "", "NA", "NA"],
    )


def test_remove_too_big_compounds(smiles_dataframe: pd.DataFrame) -> None:
    """Test nonadditivity.workflows.input_parsing:rmeove_too_big_compounds.

    Args:
        smiles_dataframe (pd.DataFrame): smiles data to check.
    """
    assert len(smiles_dataframe) == 5
    newdf = _remove_too_big_molecules(smiles_dataframe, max_heavy=60)
    assert len(newdf) == 3


def test_parse_input_file(
    paths: dict[str, Path],
    input_truths: dict[str, ArrayLike],  # pylint:disable=W0621
) -> None:
    """Test nonadditivity.workflows.input_parsing:parse_input_file.

    Args:
        paths (dict[str, Path]): paths to input file
        input_truths (dict[str, ArrayLike]): truth values for test.
    """
    dataframe, converted_property_columns = parse_input_file(
        infile=paths["test_input"],
        error_path=paths["temp_smiles"],
        property_columns=["TEST_PCHEMBL_VALUE"],
        units=["noconv"],
        max_heavy=70,
        delimiter="tab",
        series_column="Series",
    )
    with pytest.raises(ValueError):
        parse_input_file(
            infile=paths["test_input"],
            error_path=paths["temp_smiles"],
            property_columns=["FALSE COLUMN"],
            units=["noconv"],
            max_heavy=70,
            delimiter="tab",
            series_column="TEST_SERIES_ID",
        )
    assert converted_property_columns == ["TEST_PCHEMBL_VALUE"]
    assert np.array_equal(dataframe.Series.to_numpy(), input_truths["Series"])  # type: ignore
    assert np.array_equal(dataframe.index.to_numpy(), input_truths["CMPD_TEST_ID"])  # type: ignore
    assert np.array_equal(
        dataframe.TEST_PCHEMBL_VALUE.to_numpy(),  # type: ignore
        [float(value) for value in input_truths["TEST_PCHEMBL_VALUE"]],
    )
    assert_exists_and_remove(paths["temp_smiles"], assert_not_empty=False)
