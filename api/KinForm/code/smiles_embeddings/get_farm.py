import os
import sys
import pickle
import subprocess
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import pandas as pd
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ROOT
def extract_farm_embeddings(
    smiles_list: List[str],
    farm_model_name: str = "thaonguyen217/farm_molecular_representation",
    tmp_dir: str = "/tmp/farm_pipeline",
    save_path: str = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Full FARM embedding pipeline using SMILES list order.
    Output is a pickle-safe dict: original SMILES → {"cls": np.ndarray, "mean": np.ndarray}
    Assumes cleaning and FG-enhancement preserve order and length.
    """

    os.makedirs(tmp_dir, exist_ok=True)
    script_dir = "/home/msp/saleh/farm_molecular_representation/src"

    # --- Step 1: Write raw SMILES to CSV ---
    input_csv = Path(tmp_dir) / "input.csv"
    with open(input_csv, "w") as f:
        f.write("SMILES\n")
        f.writelines([sm + "\n" for sm in smiles_list])
    print(f"Input SMILES written to {input_csv}")
    # --- Step 2: Clean SMILES ---
    cleaned_csv = Path(tmp_dir) / "cleaned.csv"
    subprocess.run(
        ["python", f"{script_dir}/(1)clean_smiles.py", "--csv_data_path", str(input_csv), "--save_smiles_path", str(cleaned_csv)],
        check=True,
    )
    print(f"Cleaned SMILES saved to {cleaned_csv}")
    # --- Step 3: Generate FG-enhanced SMILES ---
    fg_pickle = Path(tmp_dir) / "fg_enhanced.pkl"
    subprocess.run(
        ["python", f"{script_dir}/(2)gen_FG_enhanced_SMILES.py", str(cleaned_csv), str(fg_pickle)],
        check=True,
    )
    print(f"FG-enhanced SMILES saved to {fg_pickle}")
    # --- Step 4: Load cleaned SMILES and FG-enhanced ---
    cleaned_df = pd.read_csv(cleaned_csv)
    cleaned_smiles = cleaned_df["SMILES"].tolist()

    with open(fg_pickle, "rb") as f:
        fg_data = pickle.load(f)

    # --- Assert that all lengths match ---
    assert len(cleaned_smiles) == len(fg_data), "Mismatch between cleaned and FG-enhanced SMILES"
    assert len(smiles_list) == len(cleaned_smiles), f"Some input SMILES were dropped or altered during cleaning,lengths: {len(smiles_list)} vs {len(cleaned_smiles)}"

    # --- Step 5: Load FARM model ---
    tokenizer = PreTrainedTokenizerFast.from_pretrained(farm_model_name)
    model = BertForMaskedLM.from_pretrained(farm_model_name)
    model.eval()
    print(f"Loaded FARM model: {farm_model_name}")
    progress_bar = tqdm(total=len(smiles_list), desc="Processing SMILES", ncols=120)
    if save_path is None:
        save_path = str(ROOT / "results/farm_embeddings/farm_embeddings.pkl")

    if Path(save_path).exists():
        with open(save_path, "rb") as f:
            result_dict = pickle.load(f)
        print(f"Loaded {len(result_dict)} existing embeddings from {save_path}")
    else:
        result_dict = {}
    new_smiles = [s for s in smiles_list if s not in result_dict]
    print(f"{len(smiles_list) - len(new_smiles)} SMILES already embedded, {len(new_smiles)} to process.")
    for orig_smiles, fg_text in zip(smiles_list, fg_data):
        if orig_smiles not in new_smiles:
            continue  # skip already embedded
        inputs = tokenizer(fg_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]
        cls_vec = hidden[:, 0, :].squeeze(0).cpu().numpy()
        mean_vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        result_dict[orig_smiles] = {"cls": cls_vec, "mean": mean_vec}
        progress_bar.update(1)

    # --- Step 7: Save safely ---
    with open(save_path, "wb") as f:
        pickle.dump(result_dict, f, protocol=4)

    return result_dict


if __name__ == "__main__":
    DATA_DIR     = ROOT
    RAW_DLKCAT     = DATA_DIR / "data/dlkcat_raw.json"
    
    import json
    import pandas as pd

    # with RAW_DLKCAT.open("r") as fp:
    #     raw = json.load(fp)

    # raw = [d for d in raw if "." not in d["Smiles"]]
    # dlkcat_smiles = [d["Smiles"] for d in raw]
    # dlkcat_unique_smiles = list(set(dlkcat_smiles))
    # print(f"Total unique DLKcat SMILES: {len(dlkcat_unique_smiles)}")
    # eitlem_path = '/home/saleh/KinForm-1/results/eitlem_smiles.pkl'
    # with open(eitlem_path, 'rb') as f:
    #     eitlem_smiles = pickle.load(f)
    # eitlem_unique_smiles = list(set(eitlem_smiles))
    # print(f"Total unique EITLEM SMILES: {len(eitlem_unique_smiles)}")
    # all_smiles = list(set(dlkcat_unique_smiles + eitlem_unique_smiles))
    # print(f"Total unique SMILES: {len(all_smiles)}")
    from pathlib import Path
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
    print(f"Total unique KM SMILES: {len(all_smiles)}")

    extract_farm_embeddings(
        smiles_list=all_smiles,
        farm_model_name="thaonguyen217/farm_molecular_representation",
        save_path=None,
    )