#!/usr/bin/env python3
"""Retrain DLKcat per CatRange fold and predict every suite set for that fold."""

import argparse
import copy
import importlib.util
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem

from benchmark_utils import make_output_dir, write_json
from suite_partition_utils import load_fold_splits, load_master_metadata


LOG10_2 = math.log10(2.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain DLKcat on CatRange folds and predict every suite set."
    )
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument("--realkcat-root", required=True, help="Path to the CatRange repository root.")
    parser.add_argument("--dlkcat-root", required=True, help="Path to the DLKcat repository root.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat"], help="DLKcat only supports kcat.")
    parser.add_argument("--embedding", default="esmc", choices=["esmc"], help="CatRange fold family.")
    parser.add_argument("--model-dir", help="Fold checkpoint root. Default: <suite-dir>/dlkcat_retrained")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="DLKcat device.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience. <=0 disables it.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation RMSE improvement.")
    parser.add_argument("--dim", type=int, default=20, help="DLKcat hidden dimension.")
    parser.add_argument("--layer-gnn", type=int, default=3, help="DLKcat GNN layers.")
    parser.add_argument("--window", type=int, default=11, help="DLKcat CNN window.")
    parser.add_argument("--layer-cnn", type=int, default=3, help="DLKcat CNN layers.")
    parser.add_argument("--layer-output", type=int, default=3, help="DLKcat MLP output layers.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--lr-decay", type=float, default=0.5, help="Learning-rate decay factor.")
    parser.add_argument("--decay-interval", type=int, default=10, help="Epoch interval for LR decay.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Adam weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--fold",
        dest="folds",
        action="append",
        type=int,
        help="Optional fold number to process. Repeatable. Default: process every fold in the suite.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Retrain models and overwrite predictions.")
    return parser.parse_args()


def _resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DLKcat requested device=cuda, but CUDA is not available in this environment.")
    return torch.device(raw_device)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    spec.loader.exec_module(module)
    return module


def _load_dlkcat_modules(dlkcat_root: Path):
    preprocess_path = dlkcat_root / "DeeplearningApproach" / "Code" / "model" / "preprocess_all.py"
    model_path = dlkcat_root / "DeeplearningApproach" / "Code" / "example" / "model.py"
    if not preprocess_path.exists():
        raise FileNotFoundError(f"DLKcat preprocess module not found: {preprocess_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"DLKcat model module not found: {model_path}")
    preprocess_mod = _load_module("dlkcat_preprocess_all", preprocess_path)
    model_mod = _load_module("dlkcat_example_model", model_path)
    return preprocess_mod, model_mod


def _reset_preprocess_dicts(preprocess_mod) -> None:
    preprocess_mod.word_dict = defaultdict(lambda: len(preprocess_mod.word_dict))
    preprocess_mod.atom_dict = defaultdict(lambda: len(preprocess_mod.atom_dict))
    preprocess_mod.bond_dict = defaultdict(lambda: len(preprocess_mod.bond_dict))
    preprocess_mod.fingerprint_dict = defaultdict(lambda: len(preprocess_mod.fingerprint_dict))
    preprocess_mod.edge_dict = defaultdict(lambda: len(preprocess_mod.edge_dict))


def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.strip()
    cleaned["smiles"] = cleaned["smiles"].astype(str).str.strip()
    cleaned["true_value"] = pd.to_numeric(cleaned["true_value"], errors="coerce")
    substrate_series = cleaned["substrate_name"] if "substrate_name" in cleaned.columns else pd.Series("", index=cleaned.index)
    cleaned["substrate_name"] = substrate_series.fillna("").astype(str)
    cleaned = cleaned.dropna(subset=["true_value"])
    cleaned = cleaned[cleaned["true_value"] > 0]
    cleaned = cleaned[cleaned["sequence"] != ""]
    cleaned = cleaned[cleaned["smiles"] != ""]
    cleaned = cleaned[~cleaned["smiles"].str.contains(r"\.", regex=True, na=False)]
    valid_mask = cleaned["smiles"].apply(lambda smi: Chem.MolFromSmiles(str(smi).strip()) is not None)
    cleaned = cleaned.loc[valid_mask].reset_index(drop=True)
    return cleaned


def _safe_split_sequence(sequence: str, ngram: int, word_dict: dict) -> np.ndarray:
    sequence = "-" + str(sequence).strip() + "="
    words = [word_dict.get(sequence[i : i + ngram], 0) for i in range(len(sequence) - ngram + 1)]
    return np.asarray(words, dtype=np.int64)


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


def _tensorize_entry(fingerprints, adjacency, words, target_log2, device: torch.device):
    target_tensor = torch.FloatTensor([target_log2]).to(device).view(1, 1)
    return (
        torch.LongTensor(fingerprints).to(device),
        torch.FloatTensor(adjacency).to(device),
        torch.LongTensor(words).to(device),
        target_tensor,
    )


def _encode_train_rows(df: pd.DataFrame, preprocess_mod, device: torch.device):
    encoded = []
    for row in df.itertuples(index=False):
        mol = Chem.AddHs(Chem.MolFromSmiles(str(row.smiles).strip()))
        atoms = preprocess_mod.create_atoms(mol)
        i_jbond_dict = preprocess_mod.create_ijbonddict(mol)
        fingerprints = preprocess_mod.extract_fingerprints(atoms, i_jbond_dict, 2)
        adjacency = preprocess_mod.create_adjacency(mol)
        words = preprocess_mod.split_sequence(str(row.sequence), 3)
        target_log2 = math.log2(float(row.true_value))
        encoded.append(_tensorize_entry(fingerprints, adjacency, words, target_log2, device))
    return encoded


def _frozen_dicts(preprocess_mod):
    return {
        "word_dict": dict(preprocess_mod.word_dict),
        "atom_dict": dict(preprocess_mod.atom_dict),
        "bond_dict": dict(preprocess_mod.bond_dict),
        "fingerprint_dict": dict(preprocess_mod.fingerprint_dict),
        "edge_dict": dict(preprocess_mod.edge_dict),
    }


def _encode_eval_rows(df: pd.DataFrame, frozen_dicts: dict, device: torch.device):
    encoded = []
    for row in df.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.smiles).strip())
        if mol is None or "." in str(row.smiles):
            encoded.append(None)
            continue
        mol = Chem.AddHs(mol)
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
        words = _safe_split_sequence(str(row.sequence), 3, frozen_dicts["word_dict"])
        target_log2 = math.log2(float(row.true_value)) if pd.notna(row.true_value) and float(row.true_value) > 0 else 0.0
        encoded.append(_tensorize_entry(fingerprints, adjacency, words, target_log2, device))
    return encoded


def _metrics_from_log2(y_true_log2, y_pred_log2):
    y_true_log10 = np.asarray(y_true_log2, dtype=float) * LOG10_2
    y_pred_log10 = np.asarray(y_pred_log2, dtype=float) * LOG10_2
    diff = y_pred_log10 - y_true_log10
    mae = float(np.mean(np.abs(diff))) if len(diff) else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else float("nan")
    if len(y_true_log10) <= 1:
        r2 = float("nan")
    else:
        ss_res = float(np.sum(np.square(diff)))
        ss_tot = float(np.sum(np.square(y_true_log10 - np.mean(y_true_log10))))
        r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _evaluate_model(model, dataset):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for fingerprints, adjacency, words, target in dataset:
            pred = model.forward((fingerprints, adjacency, words))
            y_true.append(float(target.detach().cpu().item()))
            y_pred.append(float(pred.detach().cpu().item()))
    return _metrics_from_log2(y_true, y_pred)


def _train_fold_model(args, fold: int, model_mod, train_dataset, val_dataset, n_fingerprint: int, n_word: int, device: torch.device):
    random.seed(args.seed + int(fold))
    np.random.seed(args.seed + int(fold))
    torch.manual_seed(args.seed + int(fold))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + int(fold))

    model = model_mod.KcatPrediction(
        device,
        n_fingerprint,
        n_word,
        args.dim,
        args.layer_gnn,
        args.window,
        args.layer_cnn,
        args.layer_output,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]["lr"] *= args.lr_decay

        model.train()
        random.shuffle(train_dataset)
        loss_total = 0.0
        for fingerprints, adjacency, words, target in train_dataset:
            pred = model.forward((fingerprints, adjacency, words))
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu().item())

        train_metrics = _evaluate_model(model, train_dataset)
        val_metrics = _evaluate_model(model, val_dataset)
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_total,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        print(
            f"Fold {fold} epoch {epoch:03d} | "
            f"train_rmse={train_metrics['rmse']:.4f} | val_rmse={val_metrics['rmse']:.4f}"
        )

        improved = val_metrics["rmse"] + args.min_delta < best_val_rmse
        if improved:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        elif args.patience > 0 and epoch >= args.min_epochs:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Fold {fold}: early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_rmse": best_val_rmse}


def _predict_rows(model, df: pd.DataFrame, encoded_rows):
    rows = []
    model.eval()
    with torch.no_grad():
        for row, encoded in zip(df.itertuples(index=False), encoded_rows):
            if encoded is None:
                pred_value = float("nan")
                pred_log10 = float("nan")
            else:
                fingerprints, adjacency, words, _ = encoded
                pred_log2 = float(model.forward((fingerprints, adjacency, words)).detach().cpu().item())
                pred_value = float(math.pow(2.0, pred_log2))
                pred_log10 = float(pred_log2 * LOG10_2)
            rows.append(
                {
                    "pair_id": str(row.pair_id),
                    "parameter": "kcat",
                    "model_name": "DLKcat",
                    "sequence": str(row.sequence),
                    "smiles": str(row.smiles),
                    "substrate_name": str(getattr(row, "substrate_name", "")),
                    "pred_value": pred_value,
                    "pred_log10": pred_log10,
                    "pred_unit": "s^(-1)",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    dlkcat_root = Path(args.dlkcat_root).resolve()
    realkcat_root = Path(args.realkcat_root).resolve()
    model_root = Path(args.model_dir).resolve() if args.model_dir else (suite_dir / "dlkcat_retrained").resolve()
    make_output_dir(model_root)

    preprocess_mod, model_mod = _load_dlkcat_modules(dlkcat_root)
    manifest_df = pd.read_csv(manifest_path)
    master_df = load_master_metadata(realkcat_root)
    device = _resolve_device(args.device)
    requested_folds = None if not args.folds else {int(fold) for fold in args.folds}
    print(f"DLKcat device: {device}")

    for fold in sorted(manifest_df["fold"].dropna().astype(int).unique()):
        if requested_folds is not None and int(fold) not in requested_folds:
            continue
        fold_sets = manifest_df[manifest_df["fold"].astype(int) == int(fold)].copy()
        split_frames = load_fold_splits(
            realkcat_root=realkcat_root,
            parameter=args.parameter,
            embedding=args.embedding,
            fold=int(fold),
            split_names=("train", "val"),
            master_df=master_df,
        )
        train_df = _clean_rows(split_frames["train"])
        val_df = _clean_rows(split_frames["val"])
        if train_df.empty or val_df.empty:
            raise ValueError(f"DLKcat fold {fold} has an empty train/val split after cleaning.")

        fold_dir = make_output_dir(model_root / f"fold{fold}")
        checkpoint_path = fold_dir / "model.pt"
        history_path = fold_dir / "training_history.csv"
        dicts_path = fold_dir / "encoding_dicts.pkl"
        summary_path = fold_dir / "training_summary.json"

        if args.overwrite or not checkpoint_path.exists() or not dicts_path.exists():
            _reset_preprocess_dicts(preprocess_mod)
            train_dataset = _encode_train_rows(train_df, preprocess_mod=preprocess_mod, device=device)
            frozen_dicts = _frozen_dicts(preprocess_mod)
            val_dataset = _encode_eval_rows(val_df, frozen_dicts=frozen_dicts, device=device)
            val_dataset = [row for row in val_dataset if row is not None]
            if not train_dataset or not val_dataset:
                raise ValueError(f"DLKcat fold {fold} lost all train/val rows during encoding.")

            model, history_df, summary = _train_fold_model(
                args=args,
                fold=int(fold),
                model_mod=model_mod,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                n_fingerprint=max(len(frozen_dicts["fingerprint_dict"]), 1),
                n_word=max(len(frozen_dicts["word_dict"]), 1),
                device=device,
            )
            torch.save(model.state_dict(), checkpoint_path)
            history_df.to_csv(history_path, index=False)
            with open(dicts_path, "wb") as handle:
                pickle.dump(frozen_dicts, handle)
            write_json(
                summary_path,
                {
                    "fold": int(fold),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "best_epoch": int(summary["best_epoch"]),
                    "best_val_rmse": float(summary["best_val_rmse"]),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                },
            )
        else:
            print(f"Skipping DLKcat training for fold {fold}: existing checkpoint found")
            with open(dicts_path, "rb") as handle:
                frozen_dicts = pickle.load(handle)
            model = model_mod.KcatPrediction(
                device,
                max(len(frozen_dicts["fingerprint_dict"]), 1),
                max(len(frozen_dicts["word_dict"]), 1),
                args.dim,
                args.layer_gnn,
                args.window,
                args.layer_cnn,
                args.layer_output,
            ).to(device)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)

        for row in fold_sets.itertuples(index=False):
            set_dir = Path(row.truth_csv).resolve().parent
            pred_dir = make_output_dir(set_dir / "predictions")
            standardized_output = pred_dir / "dlkcat_standardized.csv"
            raw_output = pred_dir / "dlkcat_retrained_raw.csv"
            if standardized_output.exists() and not args.overwrite:
                print(f"Skipping DLKcat prediction for {row.set_name}: existing standardized output found")
                continue

            truth_df = pd.read_csv(Path(row.truth_csv).resolve())
            eval_df = truth_df.copy()
            eval_df["sequence"] = eval_df["sequence"].astype(str)
            eval_df["smiles"] = eval_df["smiles"].astype(str)
            substrate_series = eval_df["substrate_name"] if "substrate_name" in eval_df.columns else pd.Series("", index=eval_df.index)
            eval_df["substrate_name"] = substrate_series.fillna("").astype(str)
            eval_df["true_value"] = pd.to_numeric(eval_df["true_value"], errors="coerce")
            encoded_rows = _encode_eval_rows(eval_df, frozen_dicts=frozen_dicts, device=device)
            pred_df = _predict_rows(model, eval_df, encoded_rows)
            pred_df.to_csv(standardized_output, index=False)
            pred_df.to_csv(raw_output, index=False)
            print(f"Wrote retrained DLKcat predictions for {row.set_name}: {standardized_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
