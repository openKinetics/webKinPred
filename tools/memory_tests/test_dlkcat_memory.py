#!/usr/bin/env python3
import psutil
import os
import sys
import gc
import threading
import time
import torch

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class MemoryMonitor:
    """Monitor peak memory usage during execution"""
    def __init__(self, interval=0.01):
        self.interval = interval
        self.peak_memory = 0
        self.baseline_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring memory usage"""
        self.baseline_memory = get_memory_usage()
        self.peak_memory = self.baseline_memory
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return peak memory usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.peak_memory - self.baseline_memory
        
    def _monitor_memory(self):
        """Internal method to continuously monitor memory"""
        while self.monitoring:
            current_memory = get_memory_usage()
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            time.sleep(self.interval)

def test_dlkcat_memory():
    print("Testing DLKcat memory usage...")
    
    # Add the DLKcat path to sys.path
    sys.path.insert(0, '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Code/example')
    sys.path.insert(0, '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Code')
    
    # Set environment variables
    os.environ['DLKCAT_DATA_PATH'] = '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Data'
    os.environ['DLKCAT_RESULTS_PATH'] = '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Results'
    
    # Measure initial memory
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Import and load model
    import model
    import torch
    import numpy as np
    from collections import defaultdict
    
    # Load dictionaries and model
    data_path = '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Data'
    results_path = '/home/saleh/webKinPred/api/DLKcat/DeeplearningApproach/Results'
    
    fingerprint_dict = model.load_pickle(f'{data_path}/input/fingerprint_dict.pickle')
    atom_dict = model.load_pickle(f'{data_path}/input/atom_dict.pickle')
    bond_dict = model.load_pickle(f'{data_path}/input/bond_dict.pickle')
    word_dict = model.load_pickle(f'{data_path}/input/sequence_dict.pickle')
    
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    dim = 10
    layer_gnn = 3
    window = 11
    layer_cnn = 3
    layer_output = 3
    
    # Load model
    Kcat_model = model.KcatPrediction(device, n_fingerprint, n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(device)
    Kcat_model.load_state_dict(torch.load(f'{results_path}/output/all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50', map_location=device))
    Kcat_model.eval()
    
    # Measure memory after loading model
    model_loaded_memory = get_memory_usage()
    model_load_memory = model_loaded_memory - initial_memory
    print(f"Memory after loading model: {model_loaded_memory:.2f} MB")
    print(f"Model loading memory: {model_load_memory:.2f} MB")

    # Create a maximum length sequence (10000 amino acids)
    max_sequence_list = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=(10000,))
    max_sequence = ''.join(max_sequence_list)
    test_smiles = 'CCO'  # Simple ethanol SMILES
    
    # Process the maximum sequence with memory monitoring
    gc.collect()
    
    print("Starting memory monitoring for processing...")
    memory_monitor = MemoryMonitor()
    memory_monitor.start_monitoring()

    try:
        # Split sequence into ngrams
        def split_sequence(sequence, ngram=3):
            sequence = '-' + sequence + '='
            words = []
            for i in range(len(sequence)-ngram+1):
                try:
                    words.append(word_dict[sequence[i:i+ngram]])
                except KeyError:
                    words.append(0)
            return np.array(words)
        
        # Create molecule features
        from rdkit import Chem
        mol = Chem.MolFromSmiles(test_smiles)
        
        def create_atoms(mol):
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            for a in mol.GetAromaticAtoms():
                i = a.GetIdx()
                atoms[i] = (atoms[i], 'aromatic')
            atoms = [atom_dict.get(a, 0) for a in atoms]
            return np.array(atoms)
        
        def create_ijbonddict(mol):
            i_jbond_dict = defaultdict(lambda: [])
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bond = bond_dict.get(str(b.GetBondType()), 0)
                i_jbond_dict[i].append((j, bond))
                i_jbond_dict[j].append((i, bond))
            return i_jbond_dict
        
        def extract_fingerprints(atoms, i_jbond_dict, radius=2):
            if (len(atoms) == 1) or (radius == 0):
                fingerprints = [fingerprint_dict.get(a, 0) for a in atoms]
            else:
                nodes = atoms
                i_jedge_dict = i_jbond_dict
                for _ in range(radius):
                    fingerprints = []
                    for i, j_edge in i_jedge_dict.items():
                        neighbors = [(nodes[j], edge) for j, edge in j_edge]
                        fingerprint = (nodes[i], tuple(sorted(neighbors)))
                        fingerprints.append(fingerprint_dict.get(fingerprint, 0))
                    nodes = fingerprints
            return np.array(fingerprints)
        
        def create_adjacency(mol):
            adjacency = Chem.GetAdjacencyMatrix(mol)
            return np.array(adjacency)
        
        # Process sequence and molecule
        compound = split_sequence(max_sequence)
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius=2)
        adjacency = create_adjacency(mol)
        
        # Convert to tensors and run prediction
        compound = torch.LongTensor(compound).to(device)
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        
        with torch.no_grad():
            prediction = Kcat_model.forward((fingerprints, adjacency, compound))
            
    finally:
        # Stop monitoring and get peak memory usage
        peak_processing_memory = memory_monitor.stop_monitoring()

    after_processing = get_memory_usage()
    processing_memory_final = after_processing - memory_monitor.baseline_memory
    
    print(f"Memory baseline for processing: {memory_monitor.baseline_memory:.2f} MB")
    print(f"Memory after processing: {after_processing:.2f} MB")
    print(f"Peak processing memory usage: {peak_processing_memory:.2f} MB")
    print(f"Final processing memory usage: {processing_memory_final:.2f} MB")
    
    return model_load_memory, peak_processing_memory, memory_monitor.baseline_memory

if __name__ == "__main__":
    model_mem, peak_proc_mem, baseline_mem = test_dlkcat_memory()
    total_max_ram = baseline_mem + peak_proc_mem
    print(f"\nFINAL RESULTS:")
    print(f"DLKcat - Model loading memory: {model_mem:.2f} MB")
    print(f"DLKcat - Peak processing memory increase: {peak_proc_mem:.2f} MB")
    print(f"DLKcat - TOTAL MAX RAM NEEDED: {total_max_ram:.2f} MB")
