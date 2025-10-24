import torch
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import sys 
from pathlib import Path

# Determine repository root relative to this file
# smiles_features.py is in code/utils/, so go up two levels to get to repo root
ROOT = Path(__file__).resolve().parent.parent.parent

# add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add parent dir to path
# scikit-mol fingerprint transformers
from scikit_mol.fingerprints import (
    AtomPairFingerprintTransformer,
    AvalonFingerprintTransformer,
    MACCSKeysFingerprintTransformer,
    MHFingerprintTransformer,
    TopologicalTorsionFingerprintTransformer,
)
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab
from smiles_embeddings.smiles_transformer.pretrain_trfm import TrfmSeq2seq
from smiles_embeddings.smiles_transformer.split_util import split
def smiles_transformer(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            # print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    state_dict_path = Path(__file__).resolve().parent.parent / "smiles_embeddings" / "smiles_transformer" / "trfm_12_23000.pkl"
    # if cuda is available, device='cuda', else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trfm.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=False))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X

def farm(Smiles):
    with open(ROOT / "results/farm_embeddings/farm_embeddings.pkl", "rb") as f:
        farm_dict = pickle.load(f)

    embeddings = []
    for sm in Smiles:
        if sm not in farm_dict:
            raise KeyError(f"SMILES not found in FARM embeddings: {sm}")
        sm_dict = farm_dict[sm]
        mean_emb = sm_dict['mean']
        cls_emb = sm_dict['cls']
        emb = np.concatenate([mean_emb, cls_emb], axis=-1)
        embeddings.append(emb)
        
    return np.stack(embeddings)  
    
def _smiles_to_mols(Smiles):
    """Convert a list of SMILES to RDKit Mol objects, validating each one."""
    mols = []
    for sm in Smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {sm}")
        mols.append(mol)
    return mols


def morgan_fingerprints(Smiles, radius=2, nBits=2048):
    fingerprints = []
    for sm in tqdm(Smiles, desc="Generating Morgan fingerprints", ncols=120):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {sm}")
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            arr = np.zeros((nBits,), dtype=np.float32)
            # Convert the RDKit ExplicitBitVect to a NumPy array
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
    return np.array(fingerprints)

def molformer(Smiles):
    with open(ROOT / "results/molformer_embeddings/molformer_embeddings.pkl", "rb") as f:
        molformer_dict = pickle.load(f)

    embeddings = []
    for sm in Smiles:
        if sm not in molformer_dict:
            raise KeyError(f"SMILES not found in MolFormer embeddings: {sm}")
        embeddings.append(molformer_dict[sm]["mean"])

    return np.stack(embeddings)  

def unimol(Smiles):
    """
    Fetch UniMol embeddings from precomputed dict using original input SMILES.
    mode: 'cls' or 'mean'
    Returns array of shape [L, 768]
    """
    with open(ROOT / "results/unimol_embeddings/unimol_embeddings.pkl", "rb") as f:
        unimol_dict = pickle.load(f)

    embeddings = []
    for sm in Smiles:
        if sm not in unimol_dict:
            raise KeyError(f"SMILES not found in UniMol embeddings: {sm}")
        sm_dict = unimol_dict[sm]
        mean_emb = sm_dict['mean']
        cls_emb = sm_dict['cls']
        emb = np.concatenate([mean_emb, cls_emb], axis=-1)
        embeddings.append(emb)

    return np.stack(embeddings)  
def topological_torsion_fingerprints(Smiles):
    """
    Topological-torsion fingerprint (fixed-length bit vector).
    """
    transformer = TopologicalTorsionFingerprintTransformer()
    fps = transformer.transform(_smiles_to_mols(Smiles))
    return fps.astype(np.float32)


def minhash_fingerprints(
    Smiles
):
    """
    MinHash (MHFP) fingerprint; returns a dense binary array.
    """
    transformer = MHFingerprintTransformer()
    fps = transformer.transform(_smiles_to_mols(Smiles))
    return fps.astype(np.float32)


def maccs_fingerprints(Smiles):
    """
    MACCS-keys (fixed 167-bit) fingerprint.
    """
    transformer = MACCSKeysFingerprintTransformer()  # fpSize is fixed (167)
    fps = transformer.transform(_smiles_to_mols(Smiles))
    return fps.astype(np.float32)


def atom_pair_fingerprints(Smiles):
    """
    Atom-pair fingerprint.
    """
    transformer = AtomPairFingerprintTransformer()
    fps = transformer.transform(_smiles_to_mols(Smiles))
    return fps.astype(np.float32)


def avalon_fingerprints(Smiles):
    """
    Avalon fingerprint (RDKit’s Avalon implementation).
    """
    transformer = AvalonFingerprintTransformer()
    fps = transformer.transform(_smiles_to_mols(Smiles))
    return fps.astype(np.float32)

def smiles_to_vec(Smiles, method='smiles_transformer'):
    methods = [
        'smiles_transformer', 'MFP', 'UniMol', 'FARM', 'molformer',
        'TopologicalTorsion', 'MinHash', 'MACCS', 'AtomPair', 'Avalon',
    ]

    assert method in methods, \
        f"Unknown method: {method}. Available methods: {methods}"
    method_to_func = {
        'smiles_transformer': smiles_transformer,
        'MFP': morgan_fingerprints,
        'UniMol': unimol,
        'FARM': farm,
        'molformer': molformer,
        'TopologicalTorsion' : topological_torsion_fingerprints,
        'MinHash'            : minhash_fingerprints,
        'MACCS'              : maccs_fingerprints,
        'AtomPair'           : atom_pair_fingerprints,
        'Avalon'             : avalon_fingerprints,
    }
    if method not in method_to_func:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(method_to_func.keys())}")
    return method_to_func[method](Smiles)

if __name__ == "__main__":
    # Example usage
    smiles_list = ["CCO"]
    methods = [
        'MFP', 'UniMol', 'FARM', 'molformer', 'smiles_transformer',
        'TopologicalTorsion', 'MinHash', 'MACCS', 'AtomPair', 'Avalon',
    ]
    for method in methods:
        vec = smiles_to_vec(smiles_list, method=method)[0]
        print(f"{method} vector for {smiles_list[0]}: {vec.shape}")
    raise