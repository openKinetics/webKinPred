import sys
import os
import pandas as pd
import torch
import numpy as np
import re
import pickle
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5Tokenizer, T5EncoderModel
from transformers.utils import logging
import subprocess
logging.set_verbosity_error()

SEQ_VEC_DIR = "/home/saleh/webKinPred/media/sequence_info/protT5xl_global"
PROTT5XL_MODEL_PATH = '/home/saleh/protT5_xl/prot_t5_xl_uniref50'
# seqmap CLI (SQLite resolver)
SEQMAP_PY  = "/home/saleh/webKinPredEnv/bin/python"
SEQMAP_CLI = "/home/saleh/webKinPred/tools/seqmap/main.py"
SEQMAP_DB  = "/home/saleh/webKinPred/media/sequence_info/seqmap.sqlite3"

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('/home/saleh/webKinPred/api/UniKP-main/vocab.pkl')
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('/home/saleh/webKinPred/api/UniKP-main/trfm_12_23000.pkl', map_location=torch.device('cpu')))
    trfm.eval()

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X

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

def Seq_to_vec(sequences):
    # Resolve IDs once for all sequences (duplicates included)
    ids = resolve_seq_ids_via_cli(sequences)

    vecs = []
    seqs_to_embed = []
    ids_to_embed = []

    for seq, seq_id in zip(sequences, ids):
        vec_path = os.path.join(SEQ_VEC_DIR, f"{seq_id}.npy")
        if os.path.exists(vec_path):
            vecs.append(np.load(vec_path))
        else:
            seqs_to_embed.append(seq)
            ids_to_embed.append(seq_id)

    if seqs_to_embed:
        print(f"Generating embeddings for {len(seqs_to_embed)} sequences...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        tokenizer = T5Tokenizer.from_pretrained(PROTT5XL_MODEL_PATH, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(PROTT5XL_MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.float32)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.eval()
        print("Model loaded and moved to device.")
        for seq, sid in zip(seqs_to_embed, ids_to_embed):
            spaced = ' '.join(seq)
            spaced = re.sub(r"[UZOB]", "X", spaced)
            encoded = tokenizer.batch_encode_plus([spaced], add_special_tokens=True, padding=True)
            input_ids = torch.tensor(encoded['input_ids']).to(device)
            attention_mask = torch.tensor(encoded['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            embedding = embedding.cpu().numpy()[0]
            seq_len = (attention_mask[0] == 1).sum()
            seq_vec = embedding[:seq_len - 1].mean(axis=0)
            np.save(os.path.join(SEQ_VEC_DIR, f"{sid}.npy"), seq_vec)
            vecs.append(seq_vec)

    return np.stack(vecs)

def main(input_path, output_path, task_type):
    df = pd.read_csv(input_path)
    sequences = df['Protein Sequence'].tolist()
    smiles = df['Substrate SMILES'].tolist()

    # Feature extraction
    print("Extracting SMILES features...")
    smiles_vecs = smiles_to_vec(smiles)
    print("Extracting sequence features...")
    sequence_vecs = Seq_to_vec(sequences)

    # Concatenate
    features = np.concatenate([smiles_vecs, sequence_vecs], axis=1)

    # Load trained model
    print("Loading model...")
    model_path = F'/home/saleh/webKinPred/api/UniKP-main/models/UniKP_{task_type}.pkl'
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    preds = model.predict(features)
    print(preds)
    # convert from log10 to normal scale
    preds = np.power(10, preds)
    print(preds)
    # Output
    df_out = pd.DataFrame({'Predicted Value': preds})
    df_out.to_csv(output_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_unikp.py <input_csv> <output_csv> <task_type>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    task = sys.argv[3].upper() #
    main(input_csv, output_csv, task)
