"""Parsing input in nonadditivity analysis package.

The Functions in this file handle all the file reading and input parsing for the
NonadditivityAnalysis package.
"""

import os.path
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, SaltRemover
from tqdm.auto import tqdm

from nonadditivity.utils.log import get_logger
from nonadditivity.utils.math import is_number

logger = get_logger()
DELIMITERS: dict[str, str | None] = {
    "comma": ",",
    "tab": "\t",
    "space": " ",
    "semicolon": ";",
    "": None,
}
LOG_FLAGS = [
    "pIC50",
    "pEC50",
    "pKi",
    "pKd",
    "pCC50",
    "pIC20",
    "pID50",
    "PCHEMBL",
]
DEFAULT_UNIT = ["M"]
CONVERSIONS = {
    "m": (1.0, 1.0),
    "mm": (1e-3, 3.0),
    "um": (1e-6, 6.0),
    "nm": (1e-9, 9.0),
    "pm": (1e-12, 12.0),
}


def _read_in_data(
    infile: Path,
    property_columns: list[str],
    delimiter: str = "tab",
    series_column: str | None = None,
) -> pd.DataFrame:
    """Read data from infile and converts it to a pandas Dataframe.

    It guesses the SMILES and ID column with substring search.
    It renames the columns as follows:
        > Any column containing 'SMILES' is renamed to -> 'SMILES'
        > Any column containng 'ID' or 'SRN' and that is not specified by series_column
            is renamed to -> 'ID'.
        > the column given in series_column is renamed to -> 'SERIES'

    Args:
        infile (str): file to read structures from.
        property_columns (list[str]): Names of the property columns.
        delimiter (str, optional): delimiter used in input file. Defaults to "tab".
        series_column (str | None, optional): name of the series column. Defaults to
        None.

    Raises:
        ValueError: If more than one smiles column is found in the input file.
        ValueError: If more than one column is found with id or srn in it that is not
        the series_column.

    Returns:
        pd.DataFrame: parsed dataframe.
    """
    input_dataframe = pd.read_table(
        filepath_or_buffer=infile,
        sep=DELIMITERS[delimiter],
    )
    smiles, ids = [], []

    for column in input_dataframe.columns.to_numpy():
        if "smiles" in column.lower():
            logger.info("Smiles Column found: %s", column)
            smiles.append(column)
        if "srn" in column.lower() or "id" in column.lower():
            logger.info("Identifier Column found: %s", column)
            ids.append(column)

    for property_column in property_columns:
        if property_column in input_dataframe.columns:
            logger.info("Property Column found: %s", property_column)
            continue
        raise ValueError(
            f"Given Property Column not found in input file: {property_column}",
        )

    if series_column:
        input_dataframe = input_dataframe.rename(columns={series_column: "Series"})
        logger.info("Series Column found: %s", series_column)

    if len(smiles) == 1:
        input_dataframe = input_dataframe.rename(columns={smiles[0]: "SMILES"})
    else:
        raise ValueError(
            "Please assign unambiguous names (no overlap in 'SMILES', 'ID', 'SRN'.",
        )
    input_dataframe = input_dataframe.rename(columns={ids[0]: "Compound_ID"})
    input_dataframe = input_dataframe.set_index("Compound_ID", drop=False)
    return input_dataframe


def _get_salt_remover(
    salt_file: str = os.path.join(RDConfig.RDDataDir, "Salts.txt"),
) -> SaltRemover.SaltRemover:
    """Instaciates a rdkit SaltRemover object.

    Give a path to a file with salt_definitions, otherwise the default
    Salts.txt file from rdkit is used.

    Args:
        salt_file (str, optional): Path to file with salt definitions. Defaults to
        os.path.join( RDConfig.RDDataDir, "Salts.txt", ).

    Returns:
        SaltRemover.SaltRemover: rdkit Salt Remover object.
    """
    return SaltRemover.SaltRemover(defnFilename=salt_file)


def _parse_input_smiles(
    dataframe: pd.DataFrame,
    verbose: bool = True,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Parse smiles in input file.

    Takes a dataframe and adds columns containing molecular weight and number of heavy
    atoms for the molecule specified in the smiles_column. Prior to the property
    calculation all salts are removed. The SMILES in the  smiles_column are replaced
    with SMILES not containing salt. If a SMILES could not be parsed, mol_weighs and
    num_heavies is set to -1 for that entry. Returns a list with SMILES that have
    unknown salts, and a list of smiles that could not be parsed by rdkits
    MolFromSmiles function.

    Args:
        dataframe (pd.DataFrame): Dataframe to add the properties.
        verbose (bool): wheter to display tqdm status bar. Defaults to True.

    Returns:
        tuple[list[str], ...]: lists with SMILES containing unknown salts or unparsable
        SMILES.
    """
    (new_smiles, mol_weights, num_heavies, unknown_salts, faulty_smiles, mols) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    remover = _get_salt_remover()
    for index, original_smiles in tqdm(
        zip(dataframe.index.to_numpy(), dataframe.SMILES.to_numpy()),
        total=len(dataframe),
        desc="Parsing Input SMILES",
        ascii="░▒█",
        disable=not verbose,
    ):
        try:
            smiles = original_smiles.replace("\\\\", "\\")
            mol = Chem.MolFromSmiles(  # type: ignore pylint: disable=E1101
                smiles,
                sanitize=True,
            )
            mol = remover.StripMol(mol)
            smiles = Chem.MolToSmiles(  # type: ignore pylint: disable=E1101
                mol,
                canonical=True,
            )
            if "." in smiles:
                unknown_salts.append(index)
                mols.append(-1)
                num_heavies.append(-1)
                mol_weights.append(-1)
                new_smiles.append(smiles)
                continue
            num_heavies.append(mol.GetNumHeavyAtoms())
            mol_weights.append(Descriptors.MolWt(mol))
            mols.append(mol)
            new_smiles.append(smiles)
        except Exception:  # pylint: disable=W0703
            faulty_smiles.append(index)
            num_heavies.append(-1)
            mol_weights.append(-1)
            mols.append(-1)
            new_smiles.append(original_smiles)
            continue
    dataframe["SMILES"] = new_smiles
    dataframe["Molecular_Weight"] = mol_weights
    dataframe["Num_Heavy_Atoms"] = num_heavies
    dataframe["RDKit_Molecules"] = mols
    logger.info("SMILES parsed.")
    return unknown_salts, faulty_smiles, dataframe


def _remove_unknown_and_faulty_structures(
    dataframe: pd.DataFrame,
    unknown_smiles_indices: list[str],
    faulty_smiles_indices: list[str],
    error_path: Path | None = None,
) -> pd.DataFrame:
    """Remove structures with unparsable smiles.

    Removes entries from the Dataframe with the SMILES given in unknown_smiles_indices
    and faulty_smiles_indices. All takes plac in place.

    Args:
        dataframe (pd.DataFrame): dataframe to check
        unknown_smiles_indices (list[int]): smiles of unknown structures
        faulty_smiles_indices (list[int]): smiles of faulty structures
        error_path (str): path to write erronous smiles to.
    """
    if unknown_smiles_indices:
        logger.warning(
            "%i Molecules with unknown salts were found and "
            "will be excluded from further analysis",
            len(unknown_smiles_indices),
        )
    if faulty_smiles_indices:
        logger.warning(
            "%i Molecules with invalid smiles were found and "
            "will be excluded from further analysis",
            len(faulty_smiles_indices),
        )
    if any(unknown_smiles_indices) or any(faulty_smiles_indices):
        logger.warning("Check %s for erronous entries.", str(error_path))
    if error_path:
        faulty_dataframe = dataframe[
            dataframe.index.isin(
                faulty_smiles_indices + unknown_smiles_indices,
            )
        ]
        faulty_dataframe.to_csv(
            path_or_buf=error_path,
            sep="\t",
            columns=["SMILES"],
            index=False,
            header=False,
        )
    return dataframe.drop(
        index=np.unique(
            unknown_smiles_indices + faulty_smiles_indices,  # type: ignore
        ),
    )


def _check_for_censored_values(
    dataframe: pd.DataFrame,
    property_columns: list[str],
) -> pd.DataFrame:
    """Check dataframe for censored values.

    Checks for values that begin with either *, < or >. if such a value is found, the
    sign is removed from the value and stored in a seperat CENSORED column in the
    dataframe.

    Args:
        dataframe (pd.DataFrame): Dataframe to check
        property_columns (str): property row to check
    """
    dataframe = dataframe.fillna("notanumber")
    for property_column in property_columns:
        censored = []
        for index, value in zip(
            dataframe.index.to_numpy(),
            dataframe[property_column].to_numpy(),
        ):
            if is_number(value):
                censored.append("")
                continue
            if value[0] in (">", "<", "*") and is_number(value[1:]):
                dataframe.loc[index, property_column] = value[1:]
                censored.append(value[0])
                continue
            censored.append("NA")
            dataframe.loc[index, property_column] = "0"
        dataframe[f"{property_column}_Censors"] = censored
    return dataframe


def _remove_duplicate_structures(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove structures that have equal smiles, keeping the first entry.

    Checks the dataframe for entries that have the same value in their
    SMILES COLUMN. It drops all but the first in place. It also checks
    for duplicates in the ID column, if found, an Value Error is raised.

    Args:
        dataframe (pd.DataFrame): DataFrame to check.

    Raises:
        ValueError: Thrown if two or more entries have the same ID.
    """
    dataframe = dataframe.drop_duplicates(subset="SMILES", keep="first")
    if not dataframe.index.is_unique:
        raise ValueError(
            f"Two or more entries for the same identifier: \
                {dataframe.index[np.where(dataframe.index.duplicated())[0].item()]}",
        )
    logger.info(
        (
            "Entries with duplicate SMILES were removed, "
            "the first occuring entry was kept."
        ),
    )
    return dataframe


def _check_and_convert_activities(
    dataframe: pd.DataFrame,
    property_columns: list[str],
    units: list[str | None],
) -> tuple[list[str], pd.DataFrame]:
    """Assert values possible and convert them according to units given.

    Checks dataframes property_columns column and converts, if needed, the units to
    molar and takes the negative logarithm of it. The values are stored in a new
    column. property_columns and units either have to be both str, or both lists with
    the same length.

    Args:
        dataframe (pd.DataFrame): Dataframe with the values to be converted
        property_columns (Union[str, list[str]]): Name(s) of the property columns to be
        converted.
        units (Union[str, list[str]]): Unit(s) of the values in the property column(s)

    Returns:
        list[str]: Columnames where the converted values are stored.
    """

    def convert(
        dataframe: pd.DataFrame,
        property_column: str,
        unit: str | None,
    ) -> tuple[pd.DataFrame, str]:
        """Convert values.

        Converts values in property column to molar -log10.
        unit is the original unit.

        Args:
            dataframe: (pd.DataFrame):
            property_column (str): name of column with activities
            unit (str): unit of the activities.

        Returns:
            str: name of the new column with the converted values.
        """
        try:
            dict_unit = unit.lower()  # type: ignore
        except AttributeError:
            dict_unit = "m"
        if not any(
            log_flag.lower() in property_column.lower() for log_flag in LOG_FLAGS
        ):
            dataframe = dataframe[dataframe[property_column] > 0]
            dataframe = dataframe.assign(
                **{
                    f"p{property_column}": -np.log10(
                        dataframe[property_column] * CONVERSIONS[dict_unit][0],
                    ),
                },
            )
            return dataframe, f"p{property_column}"
        if CONVERSIONS[dict_unit] != 1:
            dataframe[property_column] = (
                dataframe[property_column] + CONVERSIONS[dict_unit][1]
            )
        return dataframe, property_column

    columnames = []
    for index, property_column in enumerate(property_columns):
        dataframe[property_column] = dataframe[property_column].astype(float)
        if "pchembl" in property_column.lower():
            columnames.append(property_column)
            continue
        if units[index] == "noconv":
            columnames.append(property_column)
        else:
            dataframe, newcol = convert(
                dataframe=dataframe,
                property_column=property_column,
                unit=units[index],
            )
            columnames.append(newcol)
        dataframe = dataframe.rename(
            columns={
                f"{property_column}_Censors": f"{columnames[index]}_Censors",
            },
        )
    msg = (
        "No columns were converted"
        if np.array_equal(property_columns, columnames)
        else f"{property_columns} Columns where converted to the "
        f"following columns: {columnames}"
    )
    logger.info(msg)
    return columnames, dataframe


def _remove_too_big_molecules(
    per_compound_dataframe: pd.DataFrame,
    max_heavy: int,
) -> pd.DataFrame:
    """Remove all compounds that have num heavy atoms > max heavy.

    Args:
        per_compound_dataframe (pd.DataFrame): dataframe to remove
        compounds from
        max_heavy (int): max heavy atom threshold
    """
    before = len(per_compound_dataframe)
    per_compound_dataframe = per_compound_dataframe.loc[
        per_compound_dataframe["Num_Heavy_Atoms"] <= max_heavy
    ]
    after = len(per_compound_dataframe)
    logger.info(
        "Removing %i compounds with more than %i atoms from the dataset.",
        before - after,
        max_heavy,
    )
    return per_compound_dataframe


def parse_input_file(  # pylint: disable=R0913,W0102
    infile: Path,
    property_columns: list[str],
    error_path: Path,
    max_heavy: int,
    units: Sequence[str | None] = DEFAULT_UNIT,
    delimiter: str = "tab",
    series_column: str | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Parse and cleanup input file for nonadditivity analysis.

    Reads the infile and converts it to the dataframe. subsequently it renames the
    columns,adds molecular weight and nr of heavy atoms to the entries,
    removes duplicate entries, and removes entries where the smiles was not parseable.
    It also checks and converts any activity values to the
    negative log10 of their molar value.

    Args:
        infile (str): path to the input file.
        error_path (str): path to file, where the faulty smiles are written to.
        property_columns (list[str]): Name(s) of the property columns to be converted.
        units (Union[str, list[str]]): Unit(s) of the values in the property column(s).
        defaults to "M"
        max_heavy (int): max number of heavy atoms per ligand.
        verbose (bool, optional): True for verbose output. Defaults to None.
        delimiter (str | None, optional): delimiter in inputfile. Defaults to None.
        series_column (str | None, optional): name of series column. Defaults to None.

    Returns:
        tuple[pd.DataFrame, list[str]]: dataframe and names of the
        columns with the converted activity values.
    """
    # Read infile and convert to df
    input_dataframe = _read_in_data(
        infile=infile,
        property_columns=property_columns,
        delimiter=delimiter,
        series_column=series_column,
    )

    # add mol weight, nur heavy atoms and get list of faulty/unknown smiles
    (
        unknown_smiles_indices,
        faulty_smiles_indices,
        input_dataframe,
    ) = _parse_input_smiles(
        dataframe=input_dataframe,
        verbose=verbose,
    )

    # remove entries with said smiles from dataframe
    input_dataframe = _remove_unknown_and_faulty_structures(
        dataframe=input_dataframe,
        unknown_smiles_indices=unknown_smiles_indices,
        faulty_smiles_indices=faulty_smiles_indices,
        error_path=error_path,
    )

    input_dataframe = _check_for_censored_values(
        dataframe=input_dataframe,
        property_columns=property_columns,
    )

    # remove entries with duplicate structures
    input_dataframe = _remove_duplicate_structures(dataframe=input_dataframe)

    # check and convert activity values to molar -log10.
    conv_activity_rows, input_dataframe = _check_and_convert_activities(
        dataframe=input_dataframe,
        property_columns=property_columns,
        units=units,
    )

    input_dataframe = _remove_too_big_molecules(
        per_compound_dataframe=input_dataframe,
        max_heavy=max_heavy,
    )

    return input_dataframe, conv_activity_rows
