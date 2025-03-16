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
