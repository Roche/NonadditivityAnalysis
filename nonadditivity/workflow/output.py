"""Handle output writing for the NonadditivityAnalysis package."""

from pathlib import Path

import numpy as np
import pandas as pd

from nonadditivity.utils.log import get_logger
from nonadditivity.utils.math import mad_std, sn_medmed_std
from nonadditivity.workflow.canonicalize_output import (
    canonicalize_na_dataframe,
    get_transformation_df,
)

logger = get_logger()


def write_smiles_id_file(
    per_compound_dataframe: pd.DataFrame,
    smifile: Path,
) -> None:
    """Write input file for mmpdb fragment.

    Contains smiles and compoundid values.

    Args:
        per_compound_dataframe (pd.DataFrame): Dataframe to take the smiles from
        smifile (Union[Path, str]): path where to store the smiles.
    """
    per_compound_dataframe.to_csv(
        path_or_buf=smifile,
        sep="\t",
        columns=["SMILES", "Compound_ID"],
        index=False,
        header=False,
    )


def _write_c2c_file(
    c2c_file_path: Path,
    c2c_dataframe: pd.DataFrame,
) -> None:
    """Write file mapping circle to compounds.

    Args:
        c2c_file_path (str): path to write the file to.
        c2c_dataframe (pd.DataFrame): C2C dataframe to be written out.
    """
    c2c_dataframe.to_csv(
        path_or_buf=c2c_file_path,
        index=False,
    )


def _write_naa_file(path: Path, na_dataframe: pd.DataFrame) -> None:
    """Write the main nonadditivity output file.

    Args:
        path (str): path to file
        na_dataframe (pd.DataFrame): naa circle dataframe.
    """
    na_dataframe.to_csv(path_or_buf=path, index=False)


def _transform_per_compound_dataframe(
    per_compound_dataframe: pd.DataFrame,
    property_columns: list[str],
    include_censored: bool,
    series_column: str | None = None,
) -> pd.DataFrame:
    """Transform per compound dataframe to have a row for every property.

    Transforms the per_compound_dataframe in a way that every row only contains the
    information about one compound and one property respectively instead of one
    compound and mult. properties. It furhter removes censored values if
    include_censored == False.

    Args:
        per_compound_dataframe (pd.DataFrame): dataframe to transform
        property_columns (list[str]): name of the property columns
        include_censored (bool): whether to include censored values
        series_column (str | None, optional): Name of the series column. Defaults to
        None.

    Returns:
        pd.DataFrame: pivoted per compound data
    """
    columns = (
        [
            "Censors",
            "Nonadditivity_Pure",
            "Nonadditivity_Pure_SD",
            "Nonadditivity_Pure_Count",
            "Nonadditivity_Mixed",
            "Nonadditivity_Mixed_SD",
            "Nonadditivity_Mixed_Count",
        ]
        if series_column
        else [
            "Censors",
            "Nonadditivity",
            "Nonadditivity_SD",
            "Nonadditivity_Count",
        ]
    )

    new_pc_dataframe = pd.DataFrame()
    for column in property_columns:
        not_now = property_columns.copy()
        not_now.remove(column)
        new_df = per_compound_dataframe.drop(columns=not_now)
        new_df = new_df.dropna(subset=[column])
        new_df["Property"] = [column for _ in range(len(new_df))]
        new_df["Property_Value"] = list(new_df[column].values)
        new_df = new_df.drop(columns=column)
        for col in columns:
            new_df[col] = list(new_df[f"{column}_{col}"].values)
        to_drop = set()
        for index, censor in zip(
            new_df.index.values,
            new_df[f"{column}_Censors"].values,
        ):
            if censor == "NA":
                to_drop.add(index)
                continue
            if not include_censored and censor:
                to_drop.add(index)
        new_df = new_df.drop(index=list(to_drop))
        new_pc_dataframe = pd.concat([new_pc_dataframe, new_df])
    for column in property_columns:
        new_pc_dataframe = new_pc_dataframe.drop(
            columns=[f"{column}_{col}" for col in columns],
        )
    to_drop = set()

    return new_pc_dataframe


def _write_per_compound_file(
    path: Path,
    per_compound_dataframe: pd.DataFrame,
    property_columns: list[str],
    include_censored: bool,
    series_column: str | None = None,
) -> None:
    """Write per compound file contianing, compound, nonadditivities.

    Args:
        path (str): path to file
        per_compound_dataframe (pd.DataFrame): per compound dataframe.
        property_columns (list[str]): list containint prop column names.
        include_censored (bool): Wheter tho remove rows where censors present.
        series_column (str | None, optional): name of series column. defaults to None.
    """
    out_dataframe = _transform_per_compound_dataframe(
        per_compound_dataframe=per_compound_dataframe,
        property_columns=property_columns,
        include_censored=include_censored,
        series_column=series_column,
    )
    additional = (
        ["fused_ring_indices", "aromatic_indices"]
        if "fused_ring_indices" in out_dataframe.columns
        else []
    )
    out_dataframe = out_dataframe.drop(
        columns=["Neighbor_dict", "RDKit_Molecules"]
        + additional
        + [f"{col}_Nonadditivities" for col in property_columns],
    )
    out_dataframe = out_dataframe.rename({"Censors": "Property_Censors"})
    out_dataframe.to_csv(path_or_buf=path, index=False)


def _write_std_to_cmdline(
    property_columns: list[str],
    na_dataframe: pd.DataFrame,
    series_column: str | None = None,
) -> None:
    """Write calculated nonadditivities and std devs to the command line.

    Args:
        property_columns (list[str]): Name of the property columns in dataframe
        na_dataframe (pd.DataFrame): na_dataframe object
        series_column (str | None, optional): Name of the series_column. Defaults to
        None.
    """

    def create_logging_message(
        dataframe: pd.DataFrame,
        property_column: str,
        series: str | None = None,
    ) -> str:
        values = dataframe["Nonadditivity"].to_numpy()
        num_values = len(values)
        if num_values < 2:
            return (
                "There were not sufficient nonadditivity circles found "
                "to make a prediction on assay uncertainty "
                f"for property '{property_column}'!"
            )
        msg = (
            f"\n\nEstimated Experimental Uncertainty for property '{property_column}' "
        )
        if series:
            msg += f"and series '{series}' "
        msg += f"based on {num_values} nonadditivity circles:\n"
        msg += f"normal standard deviation: {.5* np.std(values)}\n"  # type: ignore
        msg += f"MAD standard deviation: {.5*mad_std(values)}\n"  # type: ignore
        msg += "Median of Medians standard deviation: "
        msg += str(0.5 * sn_medmed_std(values)) + "\n\n"  # type: ignore
        return msg

    if series_column:
        for property_column in property_columns:
            for series in np.unique(na_dataframe.Series.values):  # type: ignore
                out_df = na_dataframe[na_dataframe.Series == series]
                out_df = out_df[out_df.Property == property_column]
                logger.info(
                    create_logging_message(
                        dataframe=out_df,
                        property_column=property_column,
                        series=series,
                    ),
                )
        return
    for property_column in property_columns:
        out_df = na_dataframe[na_dataframe.Property == property_column]
        logger.info(
            create_logging_message(
                dataframe=out_df,
                property_column=property_column,
                series=None,
            ),
        )


def write_canonical_output(
    canonical_na_df: pd.DataFrame,
    canonical_tr_df: pd.DataFrame,
    naa_file_path: Path,
) -> None:
    """Write canonical dataframes to csv files.

    Args:
        canonical_na_df (pd.DataFrame): dataframe containing canonical circles
        canonical_tr_df (pd.DataFrame): dataframe containing canonical transformations
        naa_file_path (Path): file path for naa output dataframe.
    """
    _write_naa_file(
        path=naa_file_path.parent / "canonical_na_output.csv",
        na_dataframe=canonical_na_df,
    )
    canonical_tr_df.to_csv(
        path_or_buf=naa_file_path.parent / "canonical_transformations.csv",
        index=False,
    )


def write_output_files(
    per_compound_dataframe: pd.DataFrame,
    na_dataframe: pd.DataFrame,
    c2c_dataframe: pd.DataFrame,
    c2c_file_path: Path,
    naa_file_path: Path,
    per_compound_path: Path,
    property_columns: list[str],
    include_censored: bool,
    canonicalize: bool,
    series_column: str | None = None,
) -> None:
    """Write naa, c2c and pcp output files.

    Writes all the output files that are then needed for the spotfire
    representation of the data.

    Args:
        per_compound_dataframe (pd.DataFrame): per compound dataframe
        na_dataframe (pd.DataFrame): nonadditivity circles dataframe
        c2c_dataframe (pd.DataFrame): dataframe mapping circles to compounds
        c2c_file_path (str): path to file mapping circles and compounds
        naa_file_path (str): path to file containing nonadditivity circles
        per_compound_path (str): path to file containing per compound info
        include_censored (bool): Whether to include censored values.
        series_column (str | None, optional): series column if provided.
        property_columns (list[str]): name of the property columns used in analysis.
        canonicalize (bool): Whether to canonicalize input
    """
    _write_c2c_file(
        c2c_file_path=c2c_file_path,
        c2c_dataframe=c2c_dataframe,
    )
    _write_naa_file(path=naa_file_path, na_dataframe=na_dataframe)
    _write_per_compound_file(
        path=per_compound_path,
        per_compound_dataframe=per_compound_dataframe,
        property_columns=property_columns,
        include_censored=include_censored,
        series_column=series_column,
    )
    _write_std_to_cmdline(
        property_columns=property_columns,
        na_dataframe=na_dataframe,
        series_column=series_column,
    )

    if canonicalize:
        canonical_na_df = canonicalize_na_dataframe(na_dataframe=na_dataframe)
        canonical_tr_df = get_transformation_df(canonical_df=canonical_na_df)
        write_canonical_output(
            canonical_na_df=canonical_na_df,
            canonical_tr_df=canonical_tr_df,
            naa_file_path=naa_file_path,
        )
