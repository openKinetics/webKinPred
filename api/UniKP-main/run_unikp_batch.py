import sys
import os
import pandas as pd
import torch
import numpy as np
import re
import pickle
import gc
import math
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5Tokenizer, T5EncoderModel
from transformers.utils import logging
import subprocess
logging.set_verbosity_error()

# Use environment variables to determine paths
if os.environ.get('UNIKP_MEDIA_PATH'):
    # Docker environment
    SEQ_VEC_DIR = os.environ.get('UNIKP_MEDIA_PATH') + "/sequence_info/protT5xl_global"
    PROTT5XL_MODEL_PATH = '/app/api/UniKP-main/models/protT5_xl/prot_t5_xl_uniref50'
    SEQMAP_PY = sys.executable  # Use current Python interpreter in Docker
    SEQMAP_CLI = os.environ.get('UNIKP_TOOLS_PATH') + "/seqmap/main.py"
    SEQMAP_DB = os.environ.get('UNIKP_MEDIA_PATH') + "/sequence_info/seqmap.sqlite3"
    VOCAB_PATH = '/app/api/UniKP-main/vocab.pkl'
    TRFM_PATH = '/app/api/UniKP-main/trfm_12_23000.pkl'
else:
    # Local environment
    SEQ_VEC_DIR = "/home/saleh/webKinPred/media/sequence_info/protT5xl_global"
    PROTT5XL_MODEL_PATH = '/home/saleh/webKinPred/api/UniKP-main/models/protT5_xl/prot_t5_xl_uniref50'
    SEQMAP_PY = "/home/saleh/webKinPredEnv/bin/python"
    SEQMAP_CLI = "/home/saleh/webKinPred/tools/seqmap/main.py"
    SEQMAP_DB = "/home/saleh/webKinPred/media/sequence_info/seqmap.sqlite3"
    VOCAB_PATH = '/home/saleh/webKinPred/api/UniKP-main/vocab.pkl'
    TRFM_PATH = '/home/saleh/webKinPred/api/UniKP-main/trfm_12_23000.pkl'

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab(VOCAB_PATH)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load(TRFM_PATH, map_location=torch.device('cpu')))
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

def Seq_to_vec_single(sequence, seq_id):
    """Process a single sequence and return its vector representation."""
    vec_path = os.path.join(SEQ_VEC_DIR, f"{seq_id}.npy")
    if os.path.exists(vec_path):
        return np.load(vec_path)
    
    # Generate embedding for this sequence
    print(f"Generating embedding for sequence ID: {seq_id}")
    
    # Clean up memory before loading model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(PROTT5XL_MODEL_PATH, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(PROTT5XL_MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.float32)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        spaced = ' '.join(sequence)
        spaced = re.sub(r"[UZOB]", "X", spaced)
        
        ids = tokenizer(spaced, add_special_tokens=True, padding="longest", return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        
        embedding = embedding.last_hidden_state
        embedding = embedding.cpu().numpy()[0]
        seq_len = (attention_mask[0] == 1).sum()
        seq_vec = embedding[:seq_len - 1].mean(axis=0)
        
        # Save the embedding
        np.save(vec_path, seq_vec)
        
        # Clean up
        del model, tokenizer, embedding, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return seq_vec
    except Exception as e:
        print(f"Error generating embedding for sequence {seq_id}: {e}")
        # Clean up on error
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        raise e

def main(input_path, output_path, task_type):
    df = pd.read_csv(input_path)
    sequences = df['Protein Sequence'].tolist()
    smiles = df['Substrate SMILES'].tolist()

    # Resolve sequence IDs in batch
    seq_ids = resolve_seq_ids_via_cli(sequences)
    
    predictions = []
    total_predictions = len(sequences)
    
    # Load trained model
    print("Loading model...")
    if os.environ.get('UNIKP_MEDIA_PATH'):
        # Docker environment
        model_path = f'/app/api/UniKP-main/models/UniKP_{task_type}.pkl'
    else:
        # Local environment
        model_path = f'/home/saleh/webKinPred/api/UniKP-main/models/UniKP_{task_type}.pkl'
    
    with open(model_path, "rb") as f:
        prediction_model = pickle.load(f)

    # Process predictions one by one
    for i, (seq, smile, seq_id) in enumerate(zip(sequences, smiles, seq_ids)):
        try:
            print(f"Progress: {i+1}/{total_predictions} predictions made", flush=True)
            
            # Feature extraction for single sample
            smiles_vec = smiles_to_vec([smile])
            sequence_vec = Seq_to_vec_single(seq, seq_id)
            
            # Concatenate features
            features = np.concatenate([smiles_vec, sequence_vec.reshape(1, -1)], axis=1)
            
            # Predict
            pred = prediction_model.predict(features)[0]
            # convert from log10 to normal scale
            pred = np.power(10, pred)
            predictions.append(pred)
            
            # Clean up memory after each prediction
            del smiles_vec, sequence_vec, features
            gc.collect()
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            predictions.append(None)  # Use None for failed predictions

    # Output
    df_out = pd.DataFrame({'Predicted Value': predictions})
    df_out.to_csv(output_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_unikp_batch.py <input_csv> <output_csv> <task_type>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    task = sys.argv[3].upper()
    main(input_csv, output_csv, task)
