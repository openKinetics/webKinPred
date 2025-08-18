#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN

import os
import sys
import math
import model
import torch
import requests
import pickle
import numpy as np
from rdkit import Chem
from collections import defaultdict
import gc

# Use environment variables if available, otherwise fall back to hardcoded paths
# This allows the script to work both in Docker and local environments
data_path = os.environ.get('DLKCAT_DATA_PATH', '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Data')
results_path = os.environ.get('DLKCAT_RESULTS_PATH', '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Results')

# Print the paths being used for debugging
print(f"Using data_path: {data_path}")
print(f"Using results_path: {results_path}")
fingerprint_dict = model.load_pickle(f'{data_path}/input/fingerprint_dict.pickle')
atom_dict = model.load_pickle(f'{data_path}/input/atom_dict.pickle')
bond_dict = model.load_pickle(f'{data_path}/input/bond_dict.pickle')
edge_dict = model.load_pickle(f'{data_path}/input/edge_dict.pickle')
word_dict = model.load_pickle(f'{data_path}/input/sequence_dict.pickle')

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    # words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]

    words = list()
    for i in range(len(sequence)-ngram+1) :
        try :
            words.append(word_dict[sequence[i:i+ngram]])
        except :
            word_dict[sequence[i:i+ngram]] = 0
            words.append(word_dict[sequence[i:i+ngram]])

    return np.array(words)
    # return word_dict

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    # atom_dict = defaultdict(lambda: len(atom_dict))
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print(atoms)
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    # atoms = list()
    # for a in atoms :
    #     try: 
    #         atoms.append(atom_dict[a])
    #     except :
    #         atom_dict[a] = 0
    #         atoms.append(atom_dict[a])

    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    # bond_dict = defaultdict(lambda: len(bond_dict))
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            nodes_ = {}
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_[i] = fingerprint_dict[fingerprint]

            """Also update each edge ID considering
            its two nodes on both sides."""
            i_jedge_dict_ = {}
            for i, j_edge in i_jedge_dict.items():
                i_jedge_dict_[i] = []
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

        fingerprints = [fingerprint_dict[f] for f in nodes.values()]

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        predicted_value = self.model.forward(data)

        return predicted_value

# One method to obtain SMILES by PubChem API using the website
def get_smiles(name):
    # smiles = redis_cli.get(name)
    # if smiles is None:
    try :
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
        req = requests.get(url)
        if req.status_code != 200:
            smiles = None
        else:
            smiles = req.text.strip()
        # redis_cli.set(name, smiles, ex=None)

        # print smiles
    except :
        smiles = None

    # name_smiles[name] = smiles
    return smiles

def main() :
    name = sys.argv[1:][0]
    output_name = sys.argv[1:][1]
    # with open('./input.tsv', 'r') as infile :
    with open(name, 'r') as infile :
        lines = infile.readlines()

    fingerprint_dict = model.load_pickle(f'{data_path}/input/fingerprint_dict.pickle')
    atom_dict = model.load_pickle(f'{data_path}/input/atom_dict.pickle')
    bond_dict = model.load_pickle(f'{data_path}/input/bond_dict.pickle')
    word_dict = model.load_pickle(f'{data_path}/input/sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    radius=2
    ngram=3

    dim=10
    layer_gnn=3
    side=5
    window=11
    layer_cnn=3
    layer_output=3
    lr=1e-3
    lr_decay=0.5
    decay_interval=10
    weight_decay=1e-6
    iteration=100

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)
    # torch.manual_seed(1234)
    Kcat_model = model.KcatPrediction(device, n_fingerprint, n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(device)
    Kcat_model.load_state_dict(torch.load(f'{results_path}/output/all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50', map_location=device))
    # print(state_dict.keys())
    # model.eval()
    predictor = Predictor(Kcat_model)

    print('It\'s time to start the prediction!')
    print('-----------------------------------')

    i = 0
    total_predictions = len(lines) - 1  # Subtracting the header line
    with open(output_name, 'w+') as outfile :
        outfile.write('Substrate Name\tSubstrate SMILES\tProtein Sequence\tPredicted kcat\n')
        
        # Process lines one by one (skip header)
        for line in lines[1:] :
            line_data = line.strip().split('\t')
            name = line_data[0]
            smiles = line_data[1]
            sequence = line_data[2]
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    atoms = create_atoms(mol)
                    i_jbond_dict = create_ijbonddict(mol)
                    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
                    adjacency = create_adjacency(mol)
                    words = split_sequence(sequence,ngram)

                    fingerprints = torch.LongTensor(fingerprints).to(device)
                    adjacency = torch.FloatTensor(adjacency).to(device)
                    words = torch.LongTensor(words).to(device)

                    inputs = [fingerprints, adjacency, words]
                    print('Predicting...')
                    prediction = predictor.predict(inputs)
                    Kcat_log_value = prediction.item()
                    Kcat_value = '%.4f' % math.pow(2, Kcat_log_value)
                    
                    # Clean up tensors immediately after prediction
                    del fingerprints, adjacency, words, inputs, prediction
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                else :
                    Kcat_value = 'None'
                    smiles = 'None'
                    print('Warning: No SMILES found for', name)
                
                line_item = [name, smiles, sequence, Kcat_value]
                outfile.write('\t'.join(line_item)+'\n')
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                Kcat_value = 'None'
                line_item = [name, smiles, sequence, Kcat_value]
                outfile.write('\t'.join(line_item)+'\n')
            
            i += 1
            print(f"Progress: {i}/{total_predictions} predictions made", flush=True)

    print('Prediction Done!')


if __name__ == '__main__' :
    main()
