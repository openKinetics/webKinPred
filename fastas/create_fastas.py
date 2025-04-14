# Each section creates a FASTA file from a different dataset 
# Each should be run using the conda environment respective to the dataset

# TurNup ---------------------------------------

import pickle

path = '/home/saleh/Downloads/data/kcat_data/splits/train_df_kcat.pkl'
with open(path, 'rb') as f:
    all_sequences = pickle.load(f)

output_path = '/home/saleh/webKinPred/fastas/turnup_sequences.fasta'

with open(output_path, 'w') as fasta_file:
    for i, seq in enumerate(all_sequences['Sequence'].unique()):
        fasta_file.write(f">seq_{i}\n{seq}\n")
print(f"FASTA file created at {output_path}")

#DLKcat ---------------------------------------

import json

json_path = '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json'
output_path = '/home/saleh/webKinPred/fastas/dlkcat_sequences.fasta'

with open(json_path, 'r') as f:
    data = json.load(f)

unique_sequences = list({entry['Sequence'] for entry in data})

with open(output_path, 'w') as fasta_file:
    for i, seq in enumerate(unique_sequences):
        fasta_file.write(f">seq_{i}\n{seq}\n")
print(f"FASTA file created at {output_path}")

# EITLEM ---------------------------------------
import torch

# Load EITLEM data
kcat_train_pair = torch.load('/home/saleh/Downloads/KCATTrainPairInfo (1)')
index_seqs = torch.load('/home/saleh/Downloads/index_seq')

# Collect unique sequence indices from the training set
unique_indices = {pair[0] for pair in kcat_train_pair}

# Map indices to sequences
unique_sequences = [index_seqs[idx] for idx in unique_indices]

# Write to FASTA
output_path = '/home/saleh/webKinPred/fastas/EITLEM_sequences.fasta'
with open(output_path, 'w') as fasta_file:
    for i, seq in enumerate(unique_sequences):
        fasta_file.write(f">seq_{i}\n{seq}\n")
