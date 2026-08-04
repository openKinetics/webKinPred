#!/usr/bin/env python3
"""
Run the retrained fold-5 models (DLKcat, CatPred, UniKP, EITLEM) on the
negative-holdout examples and write fresh prediction CSVs.

Usage (from CatPred_DLKcat_Benchmark directory):
    python run_holdout_inference.py --model dlkcat --holdout-csv <path> --out-csv <path>
    python run_holdout_inference.py --model eitlem  --holdout-csv <path> --out-csv <path>
    python run_holdout_inference.py --model unikp   --holdout-csv <path> --out-csv <path>

CatPred is called as a subprocess through its own environment (see the
orchestration script run_holdout_all.sh or the notebook cell).

Input CSV columns expected: pair_id, sequence, smiles, substrate_name
Output CSV columns:          pair_id, model_name, pred_value, pred_log10, pred_unit
"""

import argparse
import hashlib
import importlib.util
import math
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent
SUITE_DIR = Path(
    "/work/ssbio/aosinuga2/Python_work/EnzymeKinetics_Manuscript_Benchmark"
    "/runs/manuscript_kcat_suite_retrained_same_split"
).resolve()
DLKCAT_RETRAINED = SUITE_DIR / "dlkcat_retrained" / "fold5"
EITLEM_MODELS = SUITE_DIR / "eitlem_models"
EITLEM_CACHE = SUITE_DIR / "eitlem_cache"
UNIKP_MODELS = SUITE_DIR / "unikp_models"
UNIKP_CACHE = SUITE_DIR / "unikp_cache"
CATPRED_CHECKPOINT = (
    SUITE_DIR / "catpred_retrained" / "fold5" / "checkpoints" / "fold_0" / "model_0" / "model.pt"
)

LOG10_2 = math.log10(2.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _canonical_seq(seq: str) -> str:
    seq = re.sub(r"\s+", "", str(seq).strip())
    return re.sub(r"[UZOB]", "X", seq)


def _seq_key(seq: str) -> str:
    return _sha1(_canonical_seq(seq))


def _smi_key(smi: str) -> str:
    return _sha1(str(smi).strip())


# ---------------------------------------------------------------------------
# DLKcat
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_dlkcat(holdout_df: pd.DataFrame, parameter: str, dlkcat_root: Path) -> pd.DataFrame:
    preprocess_mod = _load_module(
        "dlkcat_preprocess_all",
        dlkcat_root / "DeeplearningApproach" / "Code" / "model" / "preprocess_all.py",
    )
    model_mod = _load_module(
        "dlkcat_example_model",
        dlkcat_root / "DeeplearningApproach" / "Code" / "example" / "model.py",
    )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from rdkit import Chem  # noqa

    dicts_path = DLKCAT_RETRAINED / "encoding_dicts.pkl"
    checkpoint_path = DLKCAT_RETRAINED / "model.pt"
    if not dicts_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DLKcat retrained fold-5 artifacts not found in {DLKCAT_RETRAINED}"
        )

    with open(dicts_path, "rb") as fh:
        frozen_dicts = pickle.load(fh)

    def _reset_preprocess():
        preprocess_mod.word_dict = defaultdict(lambda: len(preprocess_mod.word_dict))
        preprocess_mod.atom_dict = defaultdict(lambda: len(preprocess_mod.atom_dict))
        preprocess_mod.bond_dict = defaultdict(lambda: len(preprocess_mod.bond_dict))
        preprocess_mod.fingerprint_dict = defaultdict(lambda: len(preprocess_mod.fingerprint_dict))
        preprocess_mod.edge_dict = defaultdict(lambda: len(preprocess_mod.edge_dict))

    # Restore frozen dicts into preprocess module state so encoding works
    for attr, val in frozen_dicts.items():
        setattr(preprocess_mod, attr, val)

    model = model_mod.KcatPrediction(
        device,
        max(len(frozen_dicts["fingerprint_dict"]), 1),
        max(len(frozen_dicts["word_dict"]), 1),
        20, 3, 11, 3, 3,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    def _safe_split_sequence(sequence: str, ngram: int, word_dict: dict) -> np.ndarray:
        sequence = "-" + str(sequence).strip() + "="
        return np.asarray(
            [word_dict.get(sequence[i:i + ngram], 0) for i in range(len(sequence) - ngram + 1)],
            dtype=np.int64,
        )

    def _safe_create_atoms(mol, atom_dict: dict) -> np.ndarray:
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for aromatic_atom in mol.GetAromaticAtoms():
            atoms[aromatic_atom.GetIdx()] = (atoms[aromatic_atom.GetIdx()], "aromatic")
        return np.asarray([atom_dict.get(atom, 0) for atom in atoms], dtype=np.int64)

    def _safe_create_ijbonddict(mol, bond_dict: dict):
        i_jbond_dict = defaultdict(list)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_id = bond_dict.get(str(bond.GetBondType()), 0)
            i_jbond_dict[i].append((j, bond_id))
            i_jbond_dict[j].append((i, bond_id))
        return i_jbond_dict

    def _safe_extract_fingerprints(atoms, i_jbond_dict, radius: int, fingerprint_dict: dict, edge_dict: dict) -> np.ndarray:
        if (len(atoms) == 1) or (radius == 0):
            return np.asarray([fingerprint_dict.get(atom, 0) for atom in atoms], dtype=np.int64)

        nodes = atoms
        i_jedge_dict = i_jbond_dict
        fingerprints = []
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict.get(fingerprint, 0))
            nodes = fingerprints

            next_i_jedge_dict = defaultdict(list)
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    next_edge = edge_dict.get((both_side, edge), 0)
                    next_i_jedge_dict[i].append((j, next_edge))
            i_jedge_dict = next_i_jedge_dict

        return np.asarray(fingerprints, dtype=np.int64)

    def _encode_row(sequence, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles).strip())
            if mol is None or "." in str(smiles):
                return None
            mol = Chem.AddHs(mol)
            words = _safe_split_sequence(sequence, 3, frozen_dicts["word_dict"])
            atoms = _safe_create_atoms(mol, frozen_dicts["atom_dict"])
            i_jbond_dict = _safe_create_ijbonddict(mol, frozen_dicts["bond_dict"])
            fingerprints = _safe_extract_fingerprints(
                atoms,
                i_jbond_dict,
                radius=2,
                fingerprint_dict=frozen_dicts["fingerprint_dict"],
                edge_dict=frozen_dicts["edge_dict"],
            )
            adjacency = Chem.GetAdjacencyMatrix(mol)
            words_t = torch.LongTensor(words).to(device)
            fps_t = torch.LongTensor(fingerprints).to(device)
            adj_t = torch.FloatTensor(adjacency).to(device)
            return fps_t, adj_t, words_t
        except Exception as exc:
            print(f"DLKcat encoding failed for SMILES={smiles!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    rows = []
    for _, row in holdout_df.iterrows():
        enc = _encode_row(str(row["sequence"]), str(row["smiles"]))
        if enc is None:
            pred_value, pred_log10 = float("nan"), float("nan")
        else:
            fps, adj, words = enc
            with torch.no_grad():
                pred_log2 = float(model.forward((fps, adj, words)).detach().cpu().item())
            pred_value = math.pow(2.0, pred_log2)
            pred_log10 = pred_log2 * LOG10_2
        rows.append({
            "pair_id": str(row["pair_id"]),
            "model_name": "DLKcat",
            "pred_value": pred_value,
            "pred_log10": pred_log10,
            "pred_unit": "s^(-1)",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EITLEM
# ---------------------------------------------------------------------------

def _run_eitlem(holdout_df: pd.DataFrame, parameter: str, eitlem_root: Path) -> pd.DataFrame:
    import torch
    import esm as esm_mod
    from rdkit import Chem  # noqa

    code_root = eitlem_root / "Code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from dataset import EitlemDataLoader, EitlemDataSet  # type: ignore
    from KCM import EitlemKcatPredictor  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_cache_dir = EITLEM_CACHE / "sequence_embeddings"
    seq_cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache any new sequences
    sequences = holdout_df["sequence"].astype(str).unique().tolist()
    pending = [s for s in sequences if not (seq_cache_dir / f"{_seq_key(_canonical_seq(s))}.pt").exists()]
    if pending:
        esmv_model, alphabet = esm_mod.pretrained.esm1v_t33_650M_UR90S_1()
        batch_converter = alphabet.get_batch_converter()
        esmv_model = esmv_model.to(device)
        esmv_model.eval()
        for seq in pending:
            can_seq = _canonical_seq(seq)
            batch = [("seq", can_seq)]
            _, _, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)
            lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = esmv_model(batch_tokens, repr_layers=[33], return_contacts=False)
            rep = results["representations"][33].detach().cpu()[0][1:lens[0] - 1]
            torch.save(rep, seq_cache_dir / f"{_seq_key(can_seq)}.pt")
        del esmv_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load retrained fold5 model
    model_path = EITLEM_MODELS / f"fold5_{parameter}_eitlem_core.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"EITLEM fold-5 model not found: {model_path}")
    model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pair_info = []
    smiles_lookup = {}
    for _, row in holdout_df.iterrows():
        seq_k = _seq_key(_canonical_seq(str(row["sequence"])))
        smi_k = _smi_key(str(row["smiles"]))
        smiles_lookup[smi_k] = str(row["smiles"]).strip()
        pair_info.append([seq_k, smi_k, 1.0, [str(row["pair_id"])]])

    pred_set = EitlemDataSet(pair_info, str(seq_cache_dir), smiles_lookup, 1024, 4, True, "MACCSKeys")
    pred_loader = EitlemDataLoader(
        data=pred_set, batch_size=64, shuffle=False, drop_last=False,
        num_workers=0, persistent_workers=False,
    )
    preds_log10 = []
    with torch.no_grad():
        for data in pred_loader:
            batch_preds = model(data.to(device))
            preds_log10.extend(np.asarray(batch_preds.detach().cpu()).reshape(-1).tolist())

    rows = []
    for (_, _, _, pid_list), pred_log10 in zip(pair_info, preds_log10):
        rows.append({
            "pair_id": pid_list[0],
            "model_name": "EITLEM-Kinetics",
            "pred_value": float(10.0 ** pred_log10),
            "pred_log10": float(pred_log10),
            "pred_unit": "s^(-1)",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UniKP
# ---------------------------------------------------------------------------

def _run_unikp(holdout_df: pd.DataFrame, parameter: str, unikp_root: Path) -> pd.DataFrame:
    import torch
    from sklearn.ensemble import ExtraTreesRegressor  # noqa
    from transformers import T5EncoderModel, T5Tokenizer
    try:
        from transformers.utils import import_utils as hf_import_utils
        from transformers import modeling_utils as hf_modeling_utils
        hf_import_utils.check_torch_load_is_safe = lambda: None
        hf_modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass

    if str(unikp_root) not in sys.path:
        sys.path.insert(0, str(unikp_root))
    try:
        from build_vocab import WordVocab  # type: ignore
        from pretrain_trfm import TrfmSeq2seq  # type: ignore
        from utils import split as split_smiles  # type: ignore
    except ImportError:
        import importlib
        vocab_mod = importlib.import_module("build_vocab")
        WordVocab = vocab_mod.WordVocab
        trfm_mod = importlib.import_module("pretrain_trfm")
        TrfmSeq2seq = trfm_mod.TrfmSeq2seq
        utils_mod = importlib.import_module("utils")
        split_smiles = utils_mod.split

    seq_cache_dir = UNIKP_CACHE / "sequence"
    smi_cache_dir = UNIKP_CACHE / "smiles"
    seq_cache_dir.mkdir(parents=True, exist_ok=True)
    smi_cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_seq(sequence):
        seq = re.sub(r"\s+", "", str(sequence).strip())
        return " ".join(seq)

    def _prot_t5_embed(sequences):
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        seq_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
        seq_model.eval()
        embs = {}
        for seq in sequences:
            cache_path = seq_cache_dir / f"{_sha1(seq)}.pt"
            if cache_path.exists():
                embs[seq] = torch.load(cache_path, map_location="cpu").numpy()
                continue
            inp = tokenizer([_prepare_seq(seq)], return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                out = seq_model(**inp)
            rep = out.last_hidden_state[0].mean(dim=0).detach().cpu()
            torch.save(rep, cache_path)
            embs[seq] = rep.numpy()
        del seq_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embs

    # SMILES embeddings using UniKP's own TrfmSeq2seq
    vocab_path = unikp_root / "vocab.pkl"
    trfm_path = unikp_root / "trfm_12_23000.pkl"
    if not vocab_path.exists() or not trfm_path.exists():
        raise FileNotFoundError(f"UniKP vocab/trfm files not found in {unikp_root}")

    # UniKP's distributed vocab.pkl was pickled from a script context, so the
    # pickle may look for these classes on __main__ instead of build_vocab.
    import __main__ as main_mod
    build_vocab_mod = sys.modules.get(WordVocab.__module__)
    for cls_name in ("TorchVocab", "Vocab", "WordVocab"):
        if build_vocab_mod is not None and hasattr(build_vocab_mod, cls_name):
            setattr(main_mod, cls_name, getattr(build_vocab_mod, cls_name))
    vocab = WordVocab.load_vocab(str(vocab_path))

    def _smiles_embed(smiles_list):
        embs = {}
        # Load transformer
        trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
        trfm.load_state_dict(torch.load(trfm_path, map_location="cpu"))
        trfm.eval()
        for smi in smiles_list:
            cache_path = smi_cache_dir / f"{_sha1(smi)}.pt"
            if cache_path.exists():
                embs[smi] = torch.load(cache_path, map_location="cpu").numpy()
                continue
            tokens = split_smiles(smi).split()
            if len(tokens) > 218:
                tokens = tokens[:109] + tokens[-109:]
            ids = [vocab.sos_index]
            ids.extend(vocab.stoi.get(t, vocab.unk_index) for t in tokens)
            ids.append(vocab.eos_index)
            ids.extend([vocab.pad_index] * (220 - len(ids)))
            ids_t = torch.LongTensor([ids[:220]])
            with torch.no_grad():
                out = trfm.encode(torch.t(ids_t))
            rep = np.asarray(out).squeeze()
            torch.save(torch.from_numpy(rep), cache_path)
            embs[smi] = rep
        return embs

    sequences = holdout_df["sequence"].astype(str).unique().tolist()
    smiles_list = holdout_df["smiles"].astype(str).unique().tolist()

    seq_embs = _prot_t5_embed(sequences)
    smi_embs = _smiles_embed(smiles_list)

    # Build feature matrix
    X = []
    for _, row in holdout_df.iterrows():
        se = seq_embs[str(row["sequence"])]
        sm = smi_embs[str(row["smiles"])]
        X.append(np.concatenate([se, sm]))
    X = np.array(X)

    # Load retrained model
    model_path = UNIKP_MODELS / f"fold5_{parameter}_unikp.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"UniKP fold-5 model not found: {model_path}")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    pred_log10 = model.predict(X)

    rows = []
    for (_, row), plog in zip(holdout_df.iterrows(), pred_log10):
        rows.append({
            "pair_id": str(row["pair_id"]),
            "model_name": "UniKP",
            "pred_value": float(10.0 ** float(plog)),
            "pred_log10": float(plog),
            "pred_unit": "s^(-1)",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run retrained fold-5 model on holdout inputs.")
    parser.add_argument("--model", required=True, choices=["dlkcat", "eitlem", "unikp"],
                        help="Which retrained model to run.")
    parser.add_argument("--holdout-csv", required=True,
                        help="Input CSV with pair_id, sequence, smiles, substrate_name.")
    parser.add_argument("--out-csv", required=True, help="Output predictions CSV.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km"],
                        help="Target kinetic parameter.")
    parser.add_argument("--dlkcat-root",
                        default="/work/ssbio/aosinuga2/Python_work/DLKcat",
                        help="DLKcat repository root.")
    parser.add_argument("--eitlem-root",
                        default="/work/ssbio/aosinuga2/Python_work/EITLEM-Kinetics",
                        help="EITLEM-Kinetics repository root.")
    parser.add_argument("--unikp-root",
                        default="/work/ssbio/aosinuga2/Python_work/CatPred/external/UniKP",
                        help="UniKP repository root.")
    return parser.parse_args()


def main():
    args = parse_args()
    holdout_df = pd.read_csv(args.holdout_csv)
    print(f"Running {args.model} on {len(holdout_df)} inputs …")

    if args.model == "dlkcat":
        pred_df = _run_dlkcat(holdout_df, args.parameter, Path(args.dlkcat_root))
    elif args.model == "eitlem":
        pred_df = _run_eitlem(holdout_df, args.parameter, Path(args.eitlem_root))
    elif args.model == "unikp":
        pred_df = _run_unikp(holdout_df, args.parameter, Path(args.unikp_root))
    else:
        raise ValueError(f"Unknown model: {args.model}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(pred_df)} predictions → {args.out_csv}")


if __name__ == "__main__":
    main()
