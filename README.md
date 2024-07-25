# Nonadditivity analysis

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Synposis

A program to find key complex patterns in SAR data

## Installation

The programm requires python >= 3.10.
In your already configured environment, install NonadditivityAnalysis with

```shell
pip install nonadditivity
```

### Create python environment with Conda

If you don't have an environment yet, you can use the provided (very minimal) conda environment file to create a valid environment.

```shell
conda env create -n <env_name> -f environment.yaml
conda activate <env_name>
```

or simply

```shell
conda create -n <env_name> python=3.*
```
with * being 10, 11, or 12.

Then use the `pip install nonadditivity` to install the programm.

### Dev Mode

If you want to install the package for development, some extra steps are required.

This package is managed by [Poetry](https://python-poetry.org/), so you first need to install poetry as described [here](https://python-poetry.org/docs/#installing-with-pipx).
After that, just clone the repository and install the code with poetry.

```shell
$ git clone git+https://github.com/KramerChristian/NonadditivityAnalysis.git
$ cd NonadditivityAnalysis
$ poetry install
```

## How to run the program and get help

The code runs as a simple command-line tool. Command line options are printed via

```shell
nonadditivity --help
```

## Example usage

Using the test files supplied, an example run can be

```shell
nonadditivity -i <input_file> -d <delimiter> --series-column <series_column_name> -p <property1> -p <property2> ... -u <unit1> -u <unit2>
```
or with the double-transformation cycles classification

```shell
nonadditivity -i <input_file> -d <delimiter> --series-column <series_column_name> -p <property1> -p <property2> ... -u <unit1> -u <unit2> --classify
```

### Input file format

IDENTIFIER [sep] SMILES [sep] property1 ... [sep] series_column(optional)
...

where [sep] is the separator and can be chosen from tab, space, comma, and
semicolon.

------------------

## Repo Structure

- [`examples`](example/): Contains some example input files.
- [`nonadditivity/`](nonadditivity/): Contains the source code for the package. See the [README](nonadditivity/README.md) in the folder for more info.
- [`tests`](tests/): Unit tests for the package.
- [`environment.yaml`](environment.yaml): Environment file for the conda environment.
- [`poetry.lock`](poetry.lock): File with the specification of libraries used (version and origin).
- [`pyproject.toml`](pyproject.toml): File containing build instructions for poetry as well as the metadata of the project.

## Publication

If you use this code for a publication, please cite
Kramer, C. Nonadditivity Analysis. J. Chem. Inf. Model. 2019, 59, 9, 4034–4042.
<https://pubs.acs.org/doi/10.1021/acs.jcim.9b00631>

If you are utilizing the classification module, please cite
Guasch, L., Maeder, N., Cumming, J.G., Kramer, C. From mundane to surprising nonadditivity: drivers and impact on ML models. J Comput Aided Mol Des 38, 26 (2024).
<https://doi.org/10.1007/s10822-024-00566-0>

------------------

## Background

The overall process is:

  1) Parse input:
     - read structures
     - clean and transform activity data
     - remove Salts

  2.) Compute MMPs

  3.) Find double-transformation cycles

  4.) Write to output & calculate statistics

### 1) Parse input

Ideally, the compounds are already standardized when input into nonadditivity
analysis. The code will not correct tautomers and charge state, but it will
attempt to desalt the input.

Since Nonadditivity analysis only makes sense on normally distributed data, the
input activity data can be transformed depending on the input units. You can choose
from "M", "mM", "uM", "nM", "pM", and "noconv". The 'xM' units will be transformed
to pActivity with the corresponding factors. 'noconv' keeps the input as is and does
not do any transformation.

For multiplicate structures, only the first occurence will be kept.

### 2) Compute MMPs

Matched Pairs will be computed based on the cleaned structures. This is done by a
subprocess call to the external mmpdb program. Per default, 20 parallel jobs are used
for the fragmentation. This can be changed on line 681.

### 3) Find double-transformation cycles

This is the heart of the Nonadditivity algorithm. Here, sets of four compounds that are
linked by two transformations are identified. For more details about the interpretation
see publication above.

### 4) Classify double-transformatoin cycles

Runs a bunch of classification functions that calculate topological as well as physico-chemical
properties of a double transformation cycle to help you filter out uninteresting cases when
analysing the created data. Only runs if `--classify` is provided in the command line.

### 5) Write to output and calculate statistics

Information about the compounds making up the cycles and the distribution of
nonadditivity is written to output files. [...] denotes the input file name.
The file named

`"NAA_output.csv"`

contains information about the cycles and the Probability distribution

The file named

`"perCompound.csv"`

contains information about the Nonadditivity aggregated per Compound across all cycles
where a given compound occurs.

The file named

`"c2c.csv"`

links the two files above and can be used for examnple for visualizations in SpotFire.

If you provide the `--classify` flag in the command line, `"NAA_output.csv"` and `"perCompound.csv"` will contain additional columns with the implemented descriptors.

If you provide the `--canonicalize` flag in the command line, there are two more files genrated.

The first file named

`"canonical_na_output.csv"`

is like the NAAOutput.csv, but the transformations are canonicalized, i.e. every transformation
is only occuring in one way (e.g. only "Cl>>F" and not both "Cl>>F" and "F>>Cl").

The second file named

`"canonical_transformations.csv"`

contains the transformations included here, so you can build yourself a quasi mmp analysis with this output.

------------------

## Copyright

The NonadditivityAnalysis code is copyright 2015-2024 by F. Hoffmann-La
Roche Ltd and distributed under Apache 2.0 license (see LICENSE.txt).
