import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from rdkit import Chem

from unimol_tools import UniMolRepr
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ROOT

def canonicalize(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False), True
    except:
        return smiles, False

def extract_unimol_embeddings(
    smiles_list: List[str],
    save_path: str = None,
    model_name: str = "unimolv2",
    model_size: str = "1.1B"
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract UniMol cls and mean(atom) embeddings for a list of SMILES and save to a dict:
    { original_smiles: {"cls": np.ndarray, "mean": np.ndarray} }
    """

    print(f"Loading UniMolV2 model ({model_name}, size={model_size})...")
    model = UniMolRepr(data_type="molecule", model_name=model_name, model_size=model_size, remove_hs=False)

    # Canonicalize
    canon_smiles = []
    count = 0
    for smiles in tqdm(smiles_list, desc="Canonicalizing SMILES"):
        cs, fine = canonicalize(smiles)
        canon_smiles.append(cs)
        if not fine:
            count += 1
    print(f"Canonicalized {len(smiles_list) - count} out of {len(smiles_list)} SMILES.")
    assert len(smiles_list) == len(canon_smiles)

    # Default save path (repo-relative) if not provided
    if save_path is None:
        save_path = str(ROOT / "results/unimol_embeddings/unimol_embeddings.pkl")

    # Load existing results if available
    if Path(save_path).exists():
        with open(save_path, "rb") as f:
            result_dict = pickle.load(f)
        print(f"Loaded {len(result_dict)} existing embeddings from {save_path}")
    else:
        result_dict = {}
    new_smiles = [sm for sm in smiles_list if sm not in result_dict]
    print(f"Found {len(new_smiles)} new SMILES to process.")
    pbar = tqdm(total=len(new_smiles), desc="Generating UniMol embeddings",ncols=120)
    err = 0
    for orig, canon in zip(smiles_list, canon_smiles):
        if orig in result_dict:
            continue
        try:
            pbar.update(1)
            pbar.set_postfix({"errors": err})
            reprs = model.get_repr(canon, return_atomic_reprs=True)
            cls_vec = np.array(reprs["cls_repr"][0])                          # (dim,)
            atom_mat = np.array(reprs["atomic_reprs"][0])                    # (num_atoms, dim)
            mean_vec = atom_mat.mean(axis=0)
            # if all zero, set to None
            if not (np.any(cls_vec)) and (not np.any(mean_vec)):
                result_dict[orig] = {"cls": None, "mean": None}
                err += 1
                continue
            result_dict[orig] = {"cls": cls_vec, "mean": mean_vec}
        except Exception as e:
            cls_vec = np.zeros(1536, dtype=np.float32)  # Fallback to zero vector
            mean_vec = np.zeros(1536, dtype=np.float32)
            result_dict[orig] = {"cls": cls_vec, "mean": mean_vec}
            err += 1
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result_dict, f, protocol=4)

    return result_dict


if __name__ == "__main__":
    # from pathlib import Path
    # import json

    # DATA_DIR = Path("/home/saleh/KinForm-1")
    # RAW_DLKCAT = DATA_DIR / "data/dlkcat_raw.json"

    # with RAW_DLKCAT.open("r") as fp:
    #     raw = json.load(fp)
    # raw = [d for d in raw if "." not in d["Smiles"]]
    # dlkcat_smiles = list(set(d["Smiles"] for d in raw))

    # with open("/home/saleh/KinForm-1/results/eitlem_smiles.pkl", "rb") as f:
    #     eitlem_smiles = pickle.load(f)
    # eitlem_unique = list(set(eitlem_smiles))

    # all_smiles = list(set(dlkcat_smiles + eitlem_unique))
    # from pathlib import Path
    import json
    # import numpy as np
    # KM_RAW_JSON = Path("/home/saleh/KinForm-1/data/KM_data_raw.json")
    # with KM_RAW_JSON.open("r") as fp:
    #     raw = json.load(fp)

    # raw = [d for d in raw
    #         if len(d["Sequence"]) <= 1499
    #         and "." not in d['smiles']]               
    # smiles    = [d["smiles"]                 for d in raw]
    # all_smiles = list(set(smiles))
    KM_RAW_JSON = ROOT / "data/EITLEM_data/KM/km_data.json"
    with KM_RAW_JSON.open("r") as fp:
        raw = json.load(fp)

    raw = [d for d in raw
            if len(d["sequence"]) <= 1499
            and "." not in d['smiles']]               
    smiles    = [d["smiles"]                 for d in raw]
    all_smiles = list(set(smiles))
    print(f"Total unique SMILES: {len(all_smiles)}")

    extract_unimol_embeddings(
        smiles_list=all_smiles,
        save_path=None,
    )
