import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors


@pytest.fixture()
def smiles_dataframe() -> pd.DataFrame:
    data = pd.DataFrame()
    data["SMILES"] = [
        "c1ccccc1.Cl",
        "c1ccccc1.Cl",
        "c1ccccc1.CCC",
        "c1ccccc1.Ag",
        "c1cccc1.Cl",
    ]
    data["Compound_ID"] = [f"ID{i+1}" for i in range(5)]
    data["Num_Heavy_Atoms"] = [20 * (i + 1) for i in range(5)]
    data = data.set_index("Compound_ID", drop=False)
    return data


@pytest.fixture()
def censored_dataframe() -> pd.DataFrame:
    data = pd.DataFrame()
    data["VALUES"] = ["1", "<2", ">3", "*4", "*5", "5", "<<", np.NaN]
    return data


@pytest.fixture()
def convert_dataframe() -> pd.DataFrame:
    data = pd.DataFrame()
    data["test_pchembl"] = [5, 4, 6]
    data["test_pIC50"] = [2, 1, 3]
    data["test_pm"] = [8, 7, 9]
    data["test_no_unit"] = [5, 4, 6]
    data["negative_or_zero"] = [0.0, -1, 3]
    return data


@pytest.fixture()
def duplicate_dataframe() -> pd.DataFrame:
    data = pd.DataFrame()
    data["Compound_ID"] = ["CPD1", "CPD2", "CPD3", "CPD2", "CPD4"]
    data = data.set_index("Compound_ID", drop=True)
    return data


@pytest.fixture()
def nondadditivity_dataframe() -> pd.DataFrame:
    pm_dict = {0: "pure", 1: "mixed"}
    data = pd.DataFrame()
    data["TEST_PROPERTY_Nonadditivities"] = [[(i, None) for i in range(5)]]
    data["TEST_PROPERTY_SERIES_Nonadditivities"] = [
        [(i, pm_dict[i % 2]) for i in range(5)],
    ]
    return data


@pytest.fixture()
def circles() -> list[list[str]]:
    return [["ID1", "ID2", "ID4", "ID3"], ["ID1", "ID5", "ID6", "ID4"]]


@pytest.fixture()
def per_cpd_dataframe() -> pd.DataFrame:
    dataframe = pd.DataFrame()
    dataframe["Compound_ID"] = [f"ID{i+1}" for i in range(8)]
    dataframe = dataframe.set_index("Compound_ID", drop=False)
    dataframe["SMILES"] = [
        "CC=Cc1ccc(C)nc1",
        "CC(F)=Cc1ccc(C)nc1",
        "CC=C(Cl)c1ccc(C)nc1",
        "CC(F)=C(Cl)c1ccc(C)nc1",
        "CC=Cc1cnc(C)c(O)c1",
        "CC(F)=C(Cl)c1cnc(C)c(O)c1",
        "CC(F)=C(Cl)C1=C(C)C(O)=C1",
        "CCC(F)CC",
    ]
    dataframe["TEST_PCHEMBL_VALUE"] = [4.5, 4.2, 3.7, 5.9, 3.4, 4.0, 0.5, 2.2]
    dataframe["TEST_PCHEMBL_VALUE2"] = dataframe["TEST_PCHEMBL_VALUE"]
    dataframe["Series"] = ["SERIES1" for _ in range(5)] + ["SERIES2" for _ in range(3)]
    dataframe["TEST_PCHEMBL_VALUE_Censors"] = ["" for _ in range(len(dataframe))]
    dataframe["TEST_PCHEMBL_VALUE2_Censors"] = [
        "",
        "<",
        "<",
        "",
        "",
        "",
        "",
        "",
    ]
    dataframe["RDKit_Molecules"] = [None for _ in range(len(dataframe))]
    dataframe["Molecular_Weight"] = [
        Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in dataframe.SMILES.to_numpy()
    ]
    dataframe["Num_Heavy_Atoms"] = [
        Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in dataframe.SMILES.to_numpy()
    ]
    dataframe["Neighbor_dict"] = [
        {
            "ID2": ("[*:1][H]>>[*:1]F", "[*:1]C(C)=Cc1ccc(C)nc1"),
            "ID3": ("[*:1][H]>>[*:1]Cl", "[*:1]C(=CC)c1ccc(C)nc1"),
            "ID5": ("[*:1][H]>>[*:1]O", "[*:1]c1cc(C=CC)cnc1C"),
            "ID7": (
                "[*:1]C=Cc1ccc(C)nc1>>[*:1]C1=C(C(Cl)=C(C)F)C=C1O",
                "[*:1]C",
            ),
            "ID4": ("[*:1]C=CC>>[*:1]C(Cl)=C(C)F", "[*:1]c1ccc(C)nc1"),
        },
        {
            "ID1": ("[*:1]F>>[*:1][H]", "[*:1]C(C)=Cc1ccc(C)nc1"),
            "ID3": ("[*:1]C=C(C)F>>[*:1]C(Cl)=CC", "[*:1]c1ccc(C)nc1"),
            "ID5": ("[*:1]C(F)=Cc1ccc(C)nc1>>[*:1]c1ncc(C=CC)cc1O", "[*:1]C"),
            "ID7": (
                "[*:1]C(F)=Cc1ccc(C)nc1>>[*:1]C1=C(C(Cl)=C(C)F)C=C1O",
                "[*:1]C",
            ),
            "ID4": ("[*:1][H]>>[*:1]Cl", "[*:1]C(=C(C)F)c1ccc(C)nc1"),
            "ID6": (
                "[*:1]C([*:2])=Cc1ccc([*:3])nc1>>[*:1]C([*:2])=C(Cl)c1cnc([*:3])c(O)c1",
                "[*:1]C.[*:3]C.[*:2]F",
            ),
        },
        {
            "ID1": ("[*:1]Cl>>[*:1][H]", "[*:1]C(=CC)c1ccc(C)nc1"),
            "ID2": ("[*:1]C(Cl)=CC>>[*:1]C=C(C)F", "[*:1]c1ccc(C)nc1"),
            "ID5": ("[*:1]C=C(Cl)c1ccc(C)nc1>>[*:1]c1ncc(C=CC)cc1O", "[*:1]C"),
            "ID7": (
                "[*:1]C=C(Cl)c1ccc(C)nc1>>[*:1]C1=C(C(Cl)=C(C)F)C=C1O",
                "[*:1]C",
            ),
            "ID4": ("[*:1][H]>>[*:1]F", "[*:1]C(C)=C(Cl)c1ccc(C)nc1"),
            "ID6": (
                "[*:1]C=C([*:2])c1ccc([*:3])nc1>>[*:1]C(F)=C([*:2])c1cnc([*:3])c(O)c1",
                "[*:1]C.[*:3]C.[*:2]Cl",
            ),
        },
        {
            "ID2": ("[*:1]Cl>>[*:1][H]", "[*:1]C(=C(C)F)c1ccc(C)nc1"),
            "ID3": ("[*:1]F>>[*:1][H]", "[*:1]C(C)=C(Cl)c1ccc(C)nc1"),
            "ID6": ("[*:1][H]>>[*:1]O", "[*:1]c1cc(C(Cl)=C(C)F)cnc1C"),
            "ID7": (
                "[*:1]c1ccc(C)nc1>>[*:1]C1=C(C)C(O)=C1",
                "[*:1]C(Cl)=C(C)F",
            ),
            "ID1": ("[*:1]C(Cl)=C(C)F>>[*:1]C=CC", "[*:1]c1ccc(C)nc1"),
            "ID5": (
                "[*:1]C(F)=C(Cl)c1ccc([*:2])nc1>>[*:1]C=Cc1cnc([*:2])c(O)c1",
                "[*:1]C.[*:2]C",
            ),
        },
        {
            "ID1": ("[*:1]O>>[*:1][H]", "[*:1]c1cc(C=CC)cnc1C"),
            "ID2": ("[*:1]c1ncc(C=CC)cc1O>>[*:1]C(F)=Cc1ccc(C)nc1", "[*:1]C"),
            "ID3": ("[*:1]c1ncc(C=CC)cc1O>>[*:1]C=C(Cl)c1ccc(C)nc1", "[*:1]C"),
            "ID7": (
                "[*:1]c1ncc(C=CC)cc1O>>[*:1]C1=C(C(Cl)=C(C)F)C=C1O",
                "[*:1]C",
            ),
            "ID4": (
                "[*:1]C=Cc1cnc([*:2])c(O)c1>>[*:1]C(F)=C(Cl)c1ccc([*:2])nc1",
                "[*:1]C.[*:2]C",
            ),
            "ID6": ("[*:1]C=CC>>[*:1]C(Cl)=C(C)F", "[*:1]c1cnc(C)c(O)c1"),
        },
        {
            "ID4": ("[*:1]O>>[*:1][H]", "[*:1]c1cc(C(Cl)=C(C)F)cnc1C"),
            "ID7": (
                "[*:1]c1cnc(C)c(O)c1>>[*:1]C1=C(C)C(O)=C1",
                "[*:1]C(Cl)=C(C)F",
            ),
            "ID3": (
                "[*:1]C(F)=C([*:2])c1cnc([*:3])c(O)c1>>[*:1]C=C([*:2])c1ccc([*:3])nc1",
                "[*:1]C.[*:3]C.[*:2]Cl",
            ),
            "ID2": (
                "[*:1]C([*:2])=C(Cl)c1cnc([*:3])c(O)c1>>[*:1]C([*:2])=Cc1ccc([*:3])nc1",
                "[*:1]C.[*:3]C.[*:2]F",
            ),
            "ID5": ("[*:1]C(Cl)=C(C)F>>[*:1]C=CC", "[*:1]c1cnc(C)c(O)c1"),
        },
        {
            "ID1": (
                "[*:1]C1=C(C(Cl)=C(C)F)C=C1O>>[*:1]C=Cc1ccc(C)nc1",
                "[*:1]C",
            ),
            "ID2": (
                "[*:1]C1=C(C(Cl)=C(C)F)C=C1O>>[*:1]C(F)=Cc1ccc(C)nc1",
                "[*:1]C",
            ),
            "ID3": (
                "[*:1]C1=C(C(Cl)=C(C)F)C=C1O>>[*:1]C=C(Cl)c1ccc(C)nc1",
                "[*:1]C",
            ),
            "ID5": (
                "[*:1]C1=C(C(Cl)=C(C)F)C=C1O>>[*:1]c1ncc(C=CC)cc1O",
                "[*:1]C",
            ),
            "ID4": (
                "[*:1]C1=C(C)C(O)=C1>>[*:1]c1ccc(C)nc1",
                "[*:1]C(Cl)=C(C)F",
            ),
            "ID6": (
                "[*:1]C1=C(C)C(O)=C1>>[*:1]c1cnc(C)c(O)c1",
                "[*:1]C(Cl)=C(C)F",
            ),
        },
        {},
    ]

    return dataframe


@pytest.fixture()
def same_substituent_df() -> pd.DataFrame:
    dataframe = pd.DataFrame()
    dataframe["ID"] = ["0", "1", "2", "3"]
    dataframe = dataframe.set_index(keys="ID")
    dataframe["Neighbor_dict"] = [
        {"1": ("", "[1:*]CCCCC"), "3": ("", "[1:*]CCCCC")},
        {"0": ("", "[1:*]CCCCC"), "2": ("", "[1:*]CCCCC")},
        {"1": ("", "[1:*]CCCCC"), "3": ("", "[1:*]CCCCC")},
        {"0": ("", "[1:*]CCCCC"), "2": ("", "[1:*]CCCCC")},
    ]
    return dataframe


@pytest.fixture()
def mmp_dataframe(paths: dict[str, str]) -> pd.DataFrame:
    return pd.read_table(
        filepath_or_buffer=paths["test_mmp"],
        sep="\t",
        header=None,
        names=[
            "SMILES_LHS",
            "SMILES_RHS",
            "ID_LHS",
            "ID_RHS",
            "TRANSFORMATION",
            "CONSTANT",
        ],
    )
