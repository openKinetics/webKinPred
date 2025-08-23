import os, json
import pandas as pd
from pathlib import Path
from rdkit import Chem, rdBase

try:
    from webKinPred.config_docker import KM_CSV, KCAT_CSV
except ImportError:
    from webKinPred.config_local import KM_CSV, KCAT_CSV

blocker = rdBase.BlockLogs()


def smiles_to_inchi(smiles: str) -> str | None:
    if type(smiles) is not str:
        return None
    if "InChI=" in smiles:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
    except Exception as e:
        return None
    return Chem.MolToInchi(mol) if mol else None


def lookup_experimental(
    prot_seqs: str, substrates: str, param_type: str = "Km"
) -> dict:
    """
    Quick lookup of one experimental datum.

    Parameters
    ----------
    prot_seq : str
        Full amino-acid sequence (must match exactly).
    substrate_inchi : str
        Standard InChI of the substrate.
    param_type : str
        Either "Km" or "kcat" (case-insensitive).

    Returns
    -------
    dict
        At minimum {'found': False}.
        If True, the dict contains every column present in the CSV
        for the *first* matching row.
    """

    if param_type not in {"Km", "kcat", "both"}:
        raise ValueError("param_type must be 'Km', 'kcat', or 'both'")

    if param_type == "Km":
        csv_path = KM_CSV
    elif param_type == "kcat":
        csv_path = KCAT_CSV
    else:  # both
        results_km = lookup_experimental(prot_seqs, substrates, "Km")
        results_kcat = lookup_experimental(prot_seqs, substrates, "kcat")
        # join lists of dicts
        all_results = []
        for res_km, res_kcat in zip(results_km, results_kcat):
            all_results.append(res_kcat)
            all_results.append(res_km)
        return all_results

    df = pd.read_csv(csv_path)

    results = []
    for idx, (prot_seq, substrate) in enumerate(zip(prot_seqs, substrates)):
        substrate_inchi = smiles_to_inchi(substrate)
        if substrate_inchi is None or "InChI=" not in substrate_inchi:
            results.append({"found": False, "idx": idx})
            continue
        df_subset = df[df["substrate_inchi"] == substrate_inchi]
        df_subset = df_subset[df_subset["protein_sequence"] == prot_seq]
        if df_subset.empty:
            results.append({"found": False, "idx": idx})
            continue

        row = df_subset.iloc[0]  # Get the first matching row
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        result = row.to_dict()
        result["found"] = True
        result["idx"] = idx
        results.append(result)
    return results
