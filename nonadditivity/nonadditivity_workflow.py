"""Nonadditivity Calculation Workflow."""

from nonadditivity import (
    classify,
    parse_input_file,
    run_mmpdlib_code,
    run_nonadditivity_core,
    write_output_files,
    write_smiles_id_file,
)
from nonadditivity.utils.commandline import InputOptions
from nonadditivity.utils.log import get_logger, log_versions

logger = get_logger()


@log_versions(
    logger=logger,
    packages=["mmpdblib", "rdkit", "pandas"],
    workflow="Nonadditivity Analysis",
)
def run_nonadd_calculation(input_options: InputOptions) -> None:
    """Run the Nonadditivity analysis package.

    run_control needs to have the following attributes:
    - infile (required)
    - outfile (optional)
    - property_columns (can be empty)
    - units (as many as property_columns or empty)
    - max_heavy (required)
    - update (Boolean)
    - no_chiral(Boolean)
    - write_images(Boolean)
    """
    verbose = input_options.verbose < 30

    update = input_options.update
    directory = input_options.directory
    series_column = input_options.series_column
    include_censored = input_options.include_censored

    error_path = directory / "problem_smiles.smi"
    structure_filename = directory / "ligands.smi"
    mmp_output_path = directory / "mmp_raw.csv"
    naa_file_path = directory / "NAA_output.csv"
    per_compound_path = directory / "perCompound.csv"
    c2c_file_path = directory / "c2c.csv"
    fragment_file_path = directory / "fragments.json"

    per_compound_dataframe, property_columns = parse_input_file(
        infile=input_options.infile,
        error_path=error_path,
        property_columns=input_options.property_columns,
        max_heavy=input_options.max_heavy,
        units=input_options.units,
        delimiter=input_options.delimiter,
        series_column=series_column,
        verbose=verbose,
    )

    write_smiles_id_file(
        per_compound_dataframe=per_compound_dataframe,
        smifile=structure_filename,
    )
    mmp_dataframe = run_mmpdlib_code(
        cache=update,
        fragments_file=fragment_file_path,
        mmp_outputfile=mmp_output_path,
        max_variable_heavies=input_options.max_heavy_in_transformation,
        structure_file=structure_filename,
    )
    na_dataframe, _, _, c2c_dataframe = run_nonadditivity_core(
        mmp_dataframe=mmp_dataframe,
        per_compound_dataframe=per_compound_dataframe,
        property_columns=property_columns,
        no_chiral=input_options.no_chiral,
        series_column=series_column,
        include_censored=include_censored,
        verbose=verbose,
    )
    if input_options.classify:
        _ = classify(
            per_compound_dataframe=per_compound_dataframe,
            na_dataframe=na_dataframe,
        )

    write_output_files(
        per_compound_dataframe=per_compound_dataframe,
        na_dataframe=na_dataframe,
        c2c_dataframe=c2c_dataframe,
        c2c_file_path=c2c_file_path,
        naa_file_path=naa_file_path,
        per_compound_path=per_compound_path,
        property_columns=property_columns,
        include_censored=include_censored,
        series_column=series_column,
        canonicalize=input_options.canonicalize,
    )
