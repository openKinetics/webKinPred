import yaml
import os
import logging
import sys
import pandas as pd
import numpy as np
import pickle
import argparse

from modules.utils import load_cfg
from modules.data import PocketDataset, Dataloader
from modules.TrainIters import Pseq2SitesTrainIter
from modules.helpers import *
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def main():

    """ Define argparse """
    parser = argparse.ArgumentParser(
            description = "Pseq2Sites predicts binding site based on protein sequence information"
    )
    
    parser.add_argument("--config", "-c", required = True, type = input_check, 
                    help = "The file contains information on the protein sequences to predict binding sites. \
                    (refer to the examples.inp file for a more detailed file format)")

    parser.add_argument("--labels", "-l", required = True, type = str2bool,
                        help = "labels is True: Binding site information is use for evaluation performance; \
                                labels is False: When protein' binding site information is unknwon; \
                                e.g., -t True" 
                )

    args = parser.parse_args()
    
    config = load_cfg(args.config)
    
    print("1. Load data ...")
    """ Load protein info """
    if args.labels:
        with open(config["paths"]["prot_feats"], "rb") as f:
            IDs, sequences, binding_sites, protein_feats = pickle.load(f)    

    else:
        with open(config["paths"]["prot_feats"], "rb") as f:
            IDs, sequences, protein_feats = pickle.load(f) 
    
    clean_ids, clean_seqs, clean_feats, full_seqs = [], [], [], []

    print("2. Make dataset ...")
    max_len = config["prots"]["max_lengths"]
    for prot_id, prot_seq, prot_feats in zip(IDs, sequences, protein_feats):
        full_seqs.append(prot_seq)  # always store full original sequence
        L = len(prot_seq)
        if L <= max_len:
            clean_ids.append(prot_id)
            clean_seqs.append(prot_seq)
            clean_feats.append(prot_feats)
        else:
            half = max_len // 2
            truncated_feats = np.concatenate([prot_feats[:half], prot_feats[-half:]], axis=0)
            truncated_seq = prot_seq[:half] + prot_seq[-half:]
            clean_ids.append(prot_id)
            clean_seqs.append(truncated_seq)   # for processing
            clean_feats.append(truncated_feats)


    print(f"Remove {len(IDs) - len(clean_ids)} proteins with length > {config['prots']['max_lengths']}")
    IDs, sequences, protein_feats = clean_ids, clean_seqs, clean_feats
    original_sequences = full_seqs

    dataset = PocketDataset(IDs, protein_feats, sequences)
    loader = Dataloader(dataset, batch_size = config["train"]["batch_size"], shuffle = False)

    print("3. Binding sites prediction ...")
    trainiter = Pseq2SitesTrainIter(config)
    predicted_binding_sites = trainiter.run_test(loader, config["paths"]["model_path"])

    print("4. Write predicted binding sites ...")
    if args.labels:
        fwrite(config["paths"]["result_path"], IDs, pred_binding_sites=predicted_binding_sites,
            binding_sites=binding_sites, sequences=original_sequences)
    else:
        fwrite(config["paths"]["result_path"], IDs, pred_binding_sites=predicted_binding_sites,
            sequences=original_sequences)


def fwrite(path, IDs, pred_binding_sites, binding_sites=None, sequences=None):
    """
    Write out raw per-residue binding-site probabilities.

    Args:
        path (str): output file path
        IDs (List[str]): list of protein IDs
        pred_binding_sites (ndarray): shape (N, max_len) of float probabilities
        binding_sites (List[str] or None): optional true BS strings
        sequences (List[str]): list of sequences for length info
    """
    with open(path, "w") as fw:
        # Header
        if binding_sites is not None and binding_sites[0] is not None:
            fw.write("PDB\tBS\tPred_BS_Scores\n")
            rows = zip(IDs, binding_sites, pred_binding_sites, sequences)
            for pdb_id, bs, scores, seq in rows:
                L = len(seq)
                scores_str = ",".join(f"{p:.6f}" for p in scores[:L])
                fw.write(f"{pdb_id}\t{bs}\t{scores_str}\n")
        else:
            fw.write("PDB\tPred_BS_Scores\n")
            rows = zip(IDs, pred_binding_sites, sequences)
            for pdb_id, scores, seq in rows:
                L = len(seq)
                scores_str = ",".join(f"{p:.6f}" for p in scores[:L])
                fw.write(f"{pdb_id}\t{scores_str}\n")
     
def input_check(path):

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('%s does not exist.' %path)
    return path  
    
if __name__ == "__main__":
    main()  
    