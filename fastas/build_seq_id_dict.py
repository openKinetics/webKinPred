import hashlib
import os
import json
from Bio import SeqIO
from time import time

# Paths
start = time()
fasta_paths = [
    "/home/saleh/webKinPred/fastas/dlkcat_sequences.fasta",
    "/home/saleh/webKinPred/fastas/EITLEM_sequences.fasta",
    "/home/saleh/webKinPred/fastas/turnup_sequences.fasta"
]
output_path = "/home/saleh/webKinPred/media/sequence_info/seq_id_to_seq.json"

# Helper to create unique ID
def generate_unique_seq_id(existing_ids, sequence):
    base_id = hashlib.sha256(sequence.encode()).hexdigest()[:12]
    suffix = 0
    new_id = base_id
    while new_id in existing_ids:
        suffix += 1
        new_id = f"{base_id}_{suffix}"
    return new_id

# Load and deduplicate
seq_id_dict = {}
existing_ids = set()
seen_sequences = set()

for fasta_path in fasta_paths:
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        if seq not in seen_sequences:
            seq_id = generate_unique_seq_id(existing_ids, seq)
            seq_id_dict[seq_id] = seq
            seen_sequences.add(seq)
            existing_ids.add(seq_id)

# Save as JSON
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(seq_id_dict, f, indent=2)

end = time()
print(f"Saved {len(seq_id_dict)} unique sequences to {output_path}")
print(f"Time taken: {end - start:.2f} seconds")
