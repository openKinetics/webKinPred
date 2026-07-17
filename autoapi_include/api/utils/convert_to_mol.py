from rdkit import Chem


def convert_to_mol(representation):
    """
    Converts a SMILES or InChI string to an RDKit molecule.
    Handles exceptions and invalid representations.

    Parameters:
    representation: str
        The SMILES or InChI string.

    Returns:
    RDKit Mol object or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(representation)
        if mol is None:
            mol = Chem.MolFromInchi(representation)
        return mol
    except Exception as e:
        return None


def clean_molecule_text(representation) -> str:
    if representation is None:
        return ""
    return str(representation).strip()


def is_inchi_text(representation) -> bool:
    return clean_molecule_text(representation).startswith("InChI=")


def is_smiles_text(representation) -> bool:
    text = clean_molecule_text(representation)
    if not text or is_inchi_text(text):
        return False
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def validated_molecule_text(representation) -> str | None:
    text = clean_molecule_text(representation)
    if not text:
        return None
    return text if convert_to_mol(text) is not None else None


def substrate_as_smiles(
    representation,
    *,
    canonicalize: bool = True,
    preserve_raw_smiles_when_possible: bool = False,
):
    text = clean_molecule_text(representation)
    if not text:
        return None

    mol = convert_to_mol(text)
    if mol is None:
        return None

    if not canonicalize and preserve_raw_smiles_when_possible and is_smiles_text(text):
        return text

    return Chem.MolToSmiles(mol, canonical=canonicalize)
