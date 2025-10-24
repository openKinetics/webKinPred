import argparse
import pickle
import os
from pathlib import Path
import numpy as np

seq_id_to_seq_path = "/home/msp/saleh/kinetics/results/parsed_data/sequence_id_to_sequence.pkl"
id_to_seq = pickle.load(open(seq_id_to_seq_path, "rb"))

with open('/home/msp/saleh/kinetics/results/seq_ids_recon3d.txt', 'r') as file:
    seq_ids = [line.strip() for line in file.readlines()]
seq_ids = list(set(seq_ids))

seq_dict = {sid: id_to_seq[sid] for sid in seq_ids}

features_dir = Path("/home/msp/saleh/kinetics/results/embeddings/prot_t5_last")
sequences = []
features = []
dummy_labels = []  # optional

for seq_id in seq_ids:
    seq = seq_dict[seq_id]
    feat_path = features_dir / f"{seq_id}.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature file for {seq_id}")
    feat = np.load(feat_path)

    sequences.append(seq)
    features.append(feat)

# Save output
with open('/home/msp/saleh/kinetics/results/binding_sites/pseq2sites_input/sequence_features_7.pkl', "wb") as f:
    pickle.dump((seq_ids, sequences, features), f)
