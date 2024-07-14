import pytest
from rdkit import Chem

from nonadditivity.classification import Circle, Compound, Transformation
from nonadditivity.utils.types import Molecule

# pylint: disable=redefined-outer-name


@pytest.fixture()
def compound_id1() -> str:
    return "TestID"


@pytest.fixture()
def compound_id2() -> str:
    return "TestID2"


@pytest.fixture()
def compound_id3() -> str:
    return "TestID3"


@pytest.fixture()
def compound_id4() -> str:
    return "TestID4"


@pytest.fixture()
def compound_id5() -> str:
    return "TestID5"


@pytest.fixture()
def compound_id6() -> str:
    return "TestID6"


@pytest.fixture()
def compound_id7() -> str:
    return "TestID7"


@pytest.fixture()
def compound_id8() -> str:
    return "TestID8"


@pytest.fixture()
def compound_smiles1() -> str:
    return "CCCc1ccc(Cc2ccoc2CCC)cc1C[C@@](C)(O)Cl"


@pytest.fixture()
def compound_smiles2() -> str:
    return "CCCc1ccc(Cc2ccoc2CCC)cc1C[C@@](C)(O)F"


@pytest.fixture()
def compound_smiles3() -> str:
    return "CCCc1c(C[C@H](O)C)cc(Cc2ccoc2CCC)cc1C[C@@](C)(O)F"


@pytest.fixture()
def compound_smiles4() -> str:
    return "CCCc1c(C[C@H](O)C)cc(Cc2ccoc2CCC)cc1C[C@@](C)(O)Cl"


@pytest.fixture()
def compound_smiles5() -> str:
    return "CCOc1cc(N2CCN[C@H](C)C2)ccc1C(=O)Nc1cn2cc(C)nc2cn1"


@pytest.fixture()
def compound_smiles6() -> str:
    return "Cc1cn2cc(NC(=O)c3ccc(N4CCN[C@H](C)C4)cc3F)ncc2n1"


@pytest.fixture()
def compound_smiles7() -> str:
    return "Cc1cn2cc(NC(=O)c3ccc(N4C[C@@H](C)N[C@@H](C)C4)cc3F)ncc2n1"


@pytest.fixture()
def compound_smiles8() -> str:
    return "CCOc1cc(N2C[C@@H](C)N[C@@H](C)C2)ccc1C(=O)Nc1cn2cc(C)nc2cn1"


@pytest.fixture()
def no_stereo_mol() -> Molecule:
    return Chem.MolFromSmiles("CCCC")


@pytest.fixture()
def unassigned_stereo_mol() -> Molecule:
    return Chem.MolFromSmiles("CC(Cl)(F)CCC")


@pytest.fixture()
def constant_smarts1() -> str:
    return "[*:1][C@](C)(O)Cc1cc(Cc2ccoc2CCC)ccc1CCC"


@pytest.fixture()
def constant_smarts2() -> str:
    return "[*:1]c1cc(Cc2ccoc2CCC)cc(C[C@@](C)(O)Cl)c1CCC"


@pytest.fixture()
def constant_smarts3() -> str:
    return "[*:1]c1cc(Cc2ccoc2CCC)cc(C[C@@](C)(O)F)c1CCC"


@pytest.fixture()
def constant_smarts4() -> str:
    return "[*:1][C@](C)(O)Cc1cc(Cc2ccoc2CCC)cc(C[C@H](O)C)c1CCC"


@pytest.fixture()
def constant_smarts5() -> str:
    return "[*:1]c1cc(N2CCN[C@H](C)C2)ccc1C(=O)Nc1cn2cc(C)nc2cn1"


@pytest.fixture()
def constant_smarts6() -> str:
    return "[*:1][C@H]1CN(c2ccc(C(=O)Nc3cn4cc(C)nc4cn3)c(OCC)c2)C[C@@H](C)N1"


@pytest.fixture()
def constant_smarts7() -> str:
    return "[*:1][C@H]1CN(c2ccc(C(=O)Nc3cn4cc(C)nc4cn3)c(F)c2)C[C@@H](C)N1"


@pytest.fixture()
def constant_smarts8() -> str:
    return "[*:1]c1cc(N2C[C@@H](C)N[C@@H](C)C2)ccc1C(=O)Nc1cn2cc(C)nc2cn1"


@pytest.fixture()
def transformation_smarts1() -> str:
    return "[*:1]Cl>>[*:1]F"


@pytest.fixture()
def transformation_smarts2() -> str:
    return "[*:1][H]>>[*:1]C[C@H](C)O"


@pytest.fixture()
def transformation_smarts3() -> str:
    return "[*:1]OCC>>[*:1]F"


@pytest.fixture()
def transformation_smarts4() -> str:
    return "[*:1][H]>>[*:1]C"


@pytest.fixture()
def compound_molecule1(compound_smiles1: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles1)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound_molecule2(compound_smiles2: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles2)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound_molecule3(compound_smiles3: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles3)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound_molecule4(compound_smiles4: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles4)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound_molecule5(compound_smiles5: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles5)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound_molecule6(compound_smiles6: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles6)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound_molecule7(compound_smiles7: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles7)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound_molecule8(compound_smiles8: str) -> Molecule:
    return Chem.MolFromSmiles(compound_smiles8)  # type: ignore pylint: disable=E1101


@pytest.fixture()
def compound1(
    compound_id1: str,
    compound_smiles1: str,
    compound_molecule1: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule1,
        compound_id=compound_id1,
        smiles=compound_smiles1,
    )


@pytest.fixture()
def compound2(
    compound_id2: str,
    compound_smiles2: str,
    compound_molecule2: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule2,
        compound_id=compound_id2,
        smiles=compound_smiles2,
    )


@pytest.fixture()
def compound3(
    compound_id3: str,
    compound_smiles3: str,
    compound_molecule3: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule3,
        compound_id=compound_id3,
        smiles=compound_smiles3,
    )


@pytest.fixture()
def compound4(
    compound_id4: str,
    compound_smiles4: str,
    compound_molecule4: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule4,
        compound_id=compound_id4,
        smiles=compound_smiles4,
    )


@pytest.fixture()
def compound5(
    compound_id5: str,
    compound_smiles5: str,
    compound_molecule5: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule5,
        compound_id=compound_id5,
        smiles=compound_smiles5,
    )


@pytest.fixture()
def compound6(
    compound_id6: str,
    compound_smiles6: str,
    compound_molecule6: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule6,
        compound_id=compound_id6,
        smiles=compound_smiles6,
    )


@pytest.fixture()
def compound7(
    compound_id7: str,
    compound_smiles7: str,
    compound_molecule7: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule7,
        compound_id=compound_id7,
        smiles=compound_smiles7,
    )


@pytest.fixture()
def compound8(
    compound_id8: str,
    compound_smiles8: str,
    compound_molecule8: Molecule,
) -> Compound:
    return Compound(
        molecule=compound_molecule8,
        compound_id=compound_id8,
        smiles=compound_smiles8,
    )


@pytest.fixture()
def transformation1(
    compound1: Compound,
    compound2: Compound,
    constant_smarts1: str,
    transformation_smarts1: str,
) -> Transformation:
    return Transformation(
        compound_1=compound1,
        compound_2=compound2,
        constant_smarts=constant_smarts1,
        transformation_smarts=transformation_smarts1,
    )


@pytest.fixture()
def transformation2(
    compound1: Compound,
    compound4: Compound,
    constant_smarts2: str,
    transformation_smarts2: str,
) -> Transformation:
    return Transformation(
        compound_1=compound1,
        compound_2=compound4,
        constant_smarts=constant_smarts2,
        transformation_smarts=transformation_smarts2,
    )


@pytest.fixture()
def transformation3(
    compound2: Compound,
    compound3: Compound,
    constant_smarts3: str,
    transformation_smarts2: str,
) -> Transformation:
    return Transformation(
        compound_1=compound2,
        compound_2=compound3,
        constant_smarts=constant_smarts3,
        transformation_smarts=transformation_smarts2,
    )


@pytest.fixture()
def transformation4(
    compound4: Compound,
    compound3: Compound,
    constant_smarts4: str,
    transformation_smarts1: str,
) -> Transformation:
    return Transformation(
        compound_1=compound4,
        compound_2=compound3,
        constant_smarts=constant_smarts4,
        transformation_smarts=transformation_smarts1,
    )


@pytest.fixture()
def transformation5(
    compound5: Compound,
    compound6: Compound,
    constant_smarts5: str,
    transformation_smarts3: str,
) -> Transformation:
    return Transformation(
        compound_1=compound5,
        compound_2=compound6,
        constant_smarts=constant_smarts5,
        transformation_smarts=transformation_smarts3,
    )


@pytest.fixture()
def transformation6(
    compound5: Compound,
    compound8: Compound,
    constant_smarts6: str,
    transformation_smarts4: str,
) -> Transformation:
    return Transformation(
        compound_1=compound5,
        compound_2=compound8,
        constant_smarts=constant_smarts6,
        transformation_smarts=transformation_smarts4,
    )


@pytest.fixture()
def transformation7(
    compound6: Compound,
    compound7: Compound,
    constant_smarts7: str,
    transformation_smarts4: str,
) -> Transformation:
    return Transformation(
        compound_1=compound6,
        compound_2=compound7,
        constant_smarts=constant_smarts7,
        transformation_smarts=transformation_smarts4,
    )


@pytest.fixture()
def transformation8(
    compound8: Compound,
    compound7: Compound,
    constant_smarts8: str,
    transformation_smarts3: str,
) -> Transformation:
    return Transformation(
        compound_1=compound8,
        compound_2=compound7,
        constant_smarts=constant_smarts8,
        transformation_smarts=transformation_smarts3,
    )


@pytest.fixture()
def circle(
    transformation1: Transformation,
    transformation2: Transformation,
    transformation3: Transformation,
    transformation4: Transformation,
) -> Circle:
    return Circle(
        transformation_1=transformation1,
        transformation_2=transformation2,
        transformation_3=transformation3,
        transformation_4=transformation4,
    )


@pytest.fixture()
def circle_2(
    transformation5: Transformation,
    transformation6: Transformation,
    transformation7: Transformation,
    transformation8: Transformation,
) -> Circle:
    return Circle(
        transformation_1=transformation5,
        transformation_2=transformation6,
        transformation_3=transformation7,
        transformation_4=transformation8,
    )


@pytest.fixture()
def equiv_mol() -> Molecule:
    return Chem.MolFromSmiles("Cc1cc(F)cc(F)c1")


@pytest.fixture()
def equiv_mol2() -> Molecule:
    return Chem.MolFromSmiles("CCc1cccc(CC)c1")


@pytest.fixture()
def equiv_mol3() -> Molecule:
    return Chem.MolFromSmiles("CCCC(F)(F)F")


@pytest.fixture()
def equiv_mol4() -> Molecule:
    return Chem.MolFromSmiles("COc1cc(CC(C)O)cc(OC)c1C")


@pytest.fixture()
def equiv_mol5() -> Molecule:
    return Chem.MolFromSmiles("COc1cc(CC(C)O)ccc1C")


@pytest.fixture()
def equiv_mol6() -> Molecule:
    return Chem.MolFromSmiles("Cc1ccc(CC(C)O)cc1")


@pytest.fixture()
def equiv_transformation(equiv_mol4: Molecule, equiv_mol5: Molecule) -> Transformation:
    return Transformation(
        compound_1=Compound(
            equiv_mol4,
            smiles=Chem.MolToSmiles(equiv_mol4),
            compound_id="testid",
        ),
        compound_2=Compound(
            equiv_mol5,
            smiles=Chem.MolToSmiles(equiv_mol5),
            compound_id="testid",
        ),
        transformation_smarts="[*:1]OC>>[*:1][H]",
        constant_smarts="[*:1]c1cc(CC(C)O)cc(OC)c1C",
    )


@pytest.fixture()
def equiv_transformation2(equiv_mol5: Molecule, equiv_mol6: Molecule) -> Transformation:
    return Transformation(
        compound_1=Compound(
            equiv_mol5,
            smiles=Chem.MolToSmiles(equiv_mol5),
            compound_id="testid",
        ),
        compound_2=Compound(
            equiv_mol6,
            smiles=Chem.MolToSmiles(equiv_mol6),
            compound_id="testid",
        ),
        transformation_smarts="[*:1]OC>>[*:1][H]",
        constant_smarts="[*:1]c1cc(CC(C)O)ccc1C",
    )


@pytest.fixture()
def ortho_none_circle() -> Circle:
    c1 = Compound(Chem.MolFromSmiles("c1ccccc1"), "", "")
    c2 = Compound(Chem.MolFromSmiles("c1cccc(CC)c1"), "", "")
    c3 = Compound(Chem.MolFromSmiles("c1c(CO)ccc(CC)c1"), "", "")
    c4 = Compound(Chem.MolFromSmiles("c1c(CO)cccc1"), "", "")

    return Circle(
        Transformation(c1, c2, "[*:1]c1ccccc1", "[*:1][H]>>[*:1]CC"),
        Transformation(c1, c4, "[*:1]c1ccccc1", "[*:1][H]>>[*:1]CO"),
        Transformation(c2, c3, "[*:1]c1cc(CC)ccc1", "[*:1][H]>>[*:1]CO"),
        Transformation(c4, c3, "[*:1]c1ccccc1", "[*:1][H]>>[*:1]CC"),
    )


@pytest.fixture()
def ortho_exchanged_circle() -> Circle:
    c1 = Compound(Chem.MolFromSmiles("CCc1c(CO)cccc1"), "", "")
    c2 = Compound(Chem.MolFromSmiles("CCc1c(CCl)cccc1"), "", "")
    c3 = Compound(Chem.MolFromSmiles("CCc1c(CCl)cc(F)cc1"), "", "")
    c4 = Compound(Chem.MolFromSmiles("CCc1c(CO)cc(F)cc1"), "", "")

    return Circle(
        Transformation(c1, c2, "[*:1]c1c(CC)cccc1", "[*:1]CO>>[*:1]CCl"),
        Transformation(c1, c4, "[*:1]c1cc(CO)c(CC)cc1", "[*:1][H]>>[*:1]F"),
        Transformation(c2, c3, "[*:1]c1cc(CCl)c(CC)cc1", "[*:1][H]>>[*:1]F"),
        Transformation(c4, c3, "[*:1]c1c(CC)cc(F)cc1", "[*:1]CO>>[*:1]CCl"),
    )


@pytest.fixture()
def ortho_both_circle() -> Circle:
    c1 = Compound(Chem.MolFromSmiles("CCc1c(CO)cccc1"), "", "")
    c2 = Compound(Chem.MolFromSmiles("CCc1c(CCl)cccc1"), "", "")
    c3 = Compound(Chem.MolFromSmiles("ClCc1ccccc1"), "", "")
    c4 = Compound(Chem.MolFromSmiles("OCc1ccccc1"), "", "")

    return Circle(
        Transformation(c1, c2, "[*:1]c1c(CC)cccc1", "[*:1]CO>>[*:1]CCl"),
        Transformation(c1, c4, "[*:1]c1c(CO)cccc1", "[*:1]CC>>[*:1][H]"),
        Transformation(c2, c3, "[*:1]c1c(CCl)cccc1", "[*:1]CCl>>[*:1][H]"),
        Transformation(c4, c3, "[*:1]c1ccccc1", "[*:1]CO>>[*:1]CCl"),
    )
