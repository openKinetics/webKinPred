import numpy as np
import pandas as pd
import shutil
import json
import torch
import esm
import os
from os.path import join
import hashlib

data_dir = '/home/saleh/webKinPred/api/TurNup/data'


aa = set("abcdefghiklmnpqrstxvwyzv".upper())

SEQ2ID_PATH = "/home/saleh/webKinPred/media/sequence_info/seq_id_to_seq.json"
SEQ_VEC_DIR = "/home/saleh/webKinPred/media/sequence_info/esm1b_turnup"
os.makedirs(SEQ_VEC_DIR, exist_ok=True)

# Load or initialize seq↔id mapping
if os.path.exists(SEQ2ID_PATH):
    with open(SEQ2ID_PATH, "r") as f:
        seq_id_dict = json.load(f)
else:
    seq_id_dict = {}
seq_to_id = {v: k for k, v in seq_id_dict.items()}
existing_ids = set(seq_id_dict.keys())

def generate_unique_seq_id(existing_ids, sequence):
    base_id = hashlib.sha256(sequence.encode()).hexdigest()[:12]
    suffix = 0
    new_id = base_id
    while new_id in existing_ids:
        suffix += 1
        new_id = f"{base_id}_{suffix}"
    return new_id

def validate_enzyme(seq, alphabet=aa):
    "Checks that a sequence only contains values from an alphabet"
    leftover = set(seq.upper()) - alphabet
    return not leftover

def calcualte_esm1b_ts_vectors(enzyme_list):
    df_enzyme = preprocess_enzymes(enzyme_list)

    needed_ids = []
    sequences_to_embed = []
    df_enzyme["enzyme rep"] = pd.Series([None] * len(df_enzyme), dtype=object)

    for ind in df_enzyme.index:
        seq = df_enzyme.at[ind, "model_input"]
        if seq in seq_to_id:
            seq_id = seq_to_id[seq]
        else:
            seq_id = generate_unique_seq_id(existing_ids, seq)
            seq_id_dict[seq_id] = seq
            seq_to_id[seq] = seq_id
            existing_ids.add(seq_id)
        df_enzyme.at[ind, "ID"] = seq_id

        vec_path = os.path.join(SEQ_VEC_DIR, f"{seq_id}.npy")
        if os.path.exists(vec_path):
            df_enzyme.at[ind, "enzyme rep"] = np.load(vec_path)
        else:
            sequences_to_embed.append((seq_id, seq))
            needed_ids.append(ind)

    if sequences_to_embed:
        print(f"Embedding {len(sequences_to_embed)} new sequences...")
        model_location = join(data_dir, "saved_models", "ESM1b", "esm1b_t33_650M_UR50S.pt")
        model_data = torch.load(model_location, map_location='cpu')
        regression_location = model_location[:-3] + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location='cpu')
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_data, regression_data)
        model.eval()

        batch_converter = alphabet.get_batch_converter()
        PATH = join(data_dir, "saved_models", "ESM1b", 'model_ESM_binary_A100_epoch_1_new_split.pkl')
        model_dict = torch.load(PATH, map_location='cpu')
        model_dict_V2 = {k.split("model.")[-1]: v for k, v in model_dict.items()}
        for key in ["module.fc1.weight", "module.fc1.bias", "module.fc2.weight", "module.fc2.bias", "module.fc3.weight", "module.fc3.bias"]:
            del model_dict_V2[key]
        model.load_state_dict(model_dict_V2)

        for i, (seq_id, seq) in enumerate(sequences_to_embed):
            if not validate_enzyme(seq):
                continue
            _, _, tokens = batch_converter([(seq_id, seq)])
            with torch.no_grad():
                results = model(tokens, repr_layers=[33])
            rep = results["representations"][33][0][0].numpy()
            np.save(os.path.join(SEQ_VEC_DIR, f"{seq_id}.npy"), rep)
            df_enzyme.at[needed_ids[i], "enzyme rep"] = rep

    # Save updated mapping
    with open(SEQ2ID_PATH, "w") as f:
        json.dump(seq_id_dict, f, indent=2)

    return df_enzyme


def preprocess_enzymes(enzyme_list):
	df_enzyme = pd.DataFrame(data = {"amino acid sequence" : list(set(enzyme_list))})
	df_enzyme["ID"] = ["protein_" + str(ind) for ind in df_enzyme.index]
	#if length of sequence is longer than 1020 amino acids, we crop it:
	df_enzyme["model_input"] = [seq[:1022] for seq in df_enzyme["amino acid sequence"]]
	return(df_enzyme)