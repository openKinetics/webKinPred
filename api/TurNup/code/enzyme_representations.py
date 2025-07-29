import numpy as np
import pandas as pd
import torch
import esm
import os
from os.path import join
import subprocess

data_dir = '/home/saleh/webKinPred/api/TurNup/data'

aa = set("abcdefghiklmnpqrstxvwyzv".upper())

SEQ_VEC_DIR = "/home/saleh/webKinPred/media/sequence_info/esm1b_turnup"
os.makedirs(SEQ_VEC_DIR, exist_ok=True)

# seqmap CLI (SQLite resolver)
SEQMAP_PY  = "/home/saleh/webKinPredEnv/bin/python"
SEQMAP_CLI = "/home/saleh/webKinPred/tools/seqmap/main.py"
SEQMAP_DB  = "/home/saleh/webKinPred/media/sequence_info/seqmap.sqlite3"

def resolve_seq_ids_via_cli(sequences):
    """Resolve IDs for all sequences in order (increments uses_count per occurrence)."""
    payload = "\n".join(sequences) + "\n"
    cmd = [SEQMAP_PY, SEQMAP_CLI, "--db", SEQMAP_DB, "batch-get-or-create", "--stdin"]
    proc = subprocess.run(cmd, input=payload, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"seqmap CLI failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    ids = proc.stdout.strip().splitlines()
    if len(ids) != len(sequences):
        raise RuntimeError(f"seqmap returned {len(ids)} ids for {len(sequences)} sequences")
    return ids

def validate_enzyme(seq, alphabet=aa):
    "Checks that a sequence only contains values from an alphabet"
    leftover = set(seq.upper()) - alphabet
    return not leftover

def calcualte_esm1b_ts_vectors(enzyme_list):
    df_enzyme = preprocess_enzymes(enzyme_list)

    needed_ids = []
    sequences_to_embed = []
    df_enzyme["enzyme rep"] = pd.Series([None] * len(df_enzyme), dtype=object)

    # Resolve IDs in a single call (updates uses_count & last_seen_at)
    seqs = df_enzyme["model_input"].tolist()
    ids  = resolve_seq_ids_via_cli(seqs)
    df_enzyme["ID"] = ids

    for ind in df_enzyme.index:
        seq_id = df_enzyme.at[ind, "ID"]
        seq    = df_enzyme.at[ind, "model_input"]

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

    return df_enzyme

def preprocess_enzymes(enzyme_list):
    # If you want per-occurrence counting in uses_count, remove the set():
    # df_enzyme = pd.DataFrame(data={"amino acid sequence": list(enzyme_list)})
    df_enzyme = pd.DataFrame(data = {"amino acid sequence" : list(set(enzyme_list))})
    df_enzyme["ID"] = ["protein_" + str(ind) for ind in df_enzyme.index]
    # if length of sequence is longer than 1020 amino acids, we crop it:
    df_enzyme["model_input"] = [seq[:1022] for seq in df_enzyme["amino acid sequence"]]
    return(df_enzyme)
