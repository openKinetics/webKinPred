#!/usr/bin/env python3
"""Train and run UniKP on the CatRange benchmark suite."""

import argparse
import hashlib
import os
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import make_output_dir, safe_log10
from suite_partition_utils import combine_splits, load_fold_splits, load_master_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train UniKP per CatRange fold and predict the benchmark suite.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument("--realkcat-root", required=True, help="Path to the CatRange repository root.")
    parser.add_argument("--unikp-root", required=True, help="Path to the UniKP source directory.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km"], help="Target parameter.")
    parser.add_argument("--embedding", default="esmc", choices=["esmc"], help="CatRange fold family.")
    parser.add_argument("--cache-dir", help="Embedding/model cache directory. Default: <suite-dir>/unikp_cache")
    parser.add_argument("--model-dir", help="Fold model cache directory. Default: <suite-dir>/unikp_models")
    parser.add_argument("--n-estimators", type=int, default=1000, help="ExtraTrees ensemble size.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="ExtraTrees CPU workers.")
    parser.add_argument("--seq-batch-size", type=int, default=2, help="ProtT5 embedding batch size.")
    parser.add_argument("--smiles-batch-size", type=int, default=256, help="SMILES-transformer batch size.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="ProtT5 device.")
    parser.add_argument("--exclude-val-from-train", action="store_true", help="Train on train split only.")
    parser.add_argument(
        "--fold",
        dest="folds",
        action="append",
        type=int,
        help="Optional fold number to process. Repeatable. Default: process every fold in the suite.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute fold models and predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sha1_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _cache_file(cache_root: Path, prefix: str, value: str) -> Path:
    return cache_root / prefix / f"{_sha1_token(value)}.npy"


def _prepare_sequence_text(sequence: str) -> str:
    sequence = re.sub(r"\s+", "", str(sequence).strip())
    sequence = re.sub(r"[UZOB]", "X", sequence)
    if len(sequence) > 1000:
        sequence = sequence[:500] + sequence[-500:]
    return " ".join(sequence)


def _prepare_smiles_tokens(smiles: str, split_smiles):
    return split_smiles(str(smiles).strip())


def _load_unikp_modules(unikp_root: Path):
    if str(unikp_root) not in sys.path:
        sys.path.insert(0, str(unikp_root))
    from build_vocab import WordVocab  # pylint: disable=import-error
    from pretrain_trfm import TrfmSeq2seq  # pylint: disable=import-error
    from utils import split as split_smiles  # pylint: disable=import-error

    return WordVocab, TrfmSeq2seq, split_smiles


def _resolve_device(choice: str, torch_module):
    if choice == "cpu":
        return torch_module.device("cpu")
    if choice == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("UniKP requested --device cuda, but CUDA is not available.")
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _build_smiles_encoder(unikp_root: Path, torch_module, WordVocab, TrfmSeq2seq):
    import __main__ as main_module
    import build_vocab as build_vocab_module  # pylint: disable=import-error

    vocab_path = unikp_root / "vocab.pkl"
    model_path = unikp_root / "trfm_12_23000.pkl"

    for class_name in ("TorchVocab", "Vocab", "WordVocab"):
        if hasattr(build_vocab_module, class_name):
            setattr(main_module, class_name, getattr(build_vocab_module, class_name))

    vocab = WordVocab.load_vocab(str(vocab_path))
    encoder = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    encoder.load_state_dict(torch_module.load(str(model_path), map_location="cpu"))
    encoder.eval()
    return vocab, encoder


def _smiles_ids_from_tokens(tokenized_smiles, vocab):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    seq_len = 220

    rows = []
    for tokenized in tokenized_smiles:
        tokens = tokenized.split()
        if len(tokens) > 218:
            tokens = tokens[:109] + tokens[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = [sos_index] + ids + [eos_index]
        if len(ids) < seq_len:
            ids.extend([pad_index] * (seq_len - len(ids)))
        else:
            ids = ids[:seq_len]
        rows.append(ids)
    return rows


def _cache_smiles_embeddings(
    smiles_values,
    cache_root: Path,
    split_smiles,
    vocab,
    smiles_encoder,
    torch_module,
    batch_size: int,
):
    smiles_values = [str(value) for value in smiles_values]
    pending = []
    for value in smiles_values:
        cache_path = _cache_file(cache_root, "smiles", value)
        if not cache_path.exists():
            pending.append(value)
    if not pending:
        return

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        tokenized = [_prepare_smiles_tokens(value, split_smiles) for value in chunk]
        ids = _smiles_ids_from_tokens(tokenized, vocab)
        src = torch_module.tensor(ids, dtype=torch_module.long).T
        chunk_embeddings = smiles_encoder.encode(src)
        for value, emb in zip(chunk, chunk_embeddings):
            cache_path = _cache_file(cache_root, "smiles", value)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, np.asarray(emb, dtype=np.float32))


def _cache_sequence_embeddings(
    sequences,
    cache_root: Path,
    tokenizer,
    model,
    torch_module,
    device,
    batch_size: int,
):
    sequences = [str(value) for value in sequences]
    pending = []
    for value in sequences:
        cache_path = _cache_file(cache_root, "sequence", value)
        if not cache_path.exists():
            pending.append(value)
    if not pending:
        return

    prepared = [_prepare_sequence_text(seq) for seq in pending]
    for start in range(0, len(prepared), batch_size):
        chunk_sequences = prepared[start:start + batch_size]
        raw_sequences = pending[start:start + batch_size]
        encoded = tokenizer.batch_encode_plus(
            chunk_sequences,
            add_special_tokens=True,
            padding=True,
        )
        input_ids = torch_module.tensor(encoded["input_ids"]).to(device)
        attention_mask = torch_module.tensor(encoded["attention_mask"]).to(device)
        with torch_module.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        mask_np = attention_mask.detach().cpu().numpy()

        for raw_sequence, emb, mask in zip(raw_sequences, embeddings, mask_np):
            seq_len = int(mask.sum())
            seq_emb = emb[: max(seq_len - 1, 1)]
            pooled = seq_emb.mean(axis=0).astype(np.float32)
            cache_path = _cache_file(cache_root, "sequence", raw_sequence)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, pooled)


def _load_feature_matrix(df: pd.DataFrame, cache_root: Path) -> np.ndarray:
    seq_embs = [np.load(_cache_file(cache_root, "sequence", str(seq))) for seq in df["sequence"].astype(str)]
    smiles_embs = [np.load(_cache_file(cache_root, "smiles", str(smi))) for smi in df["smiles"].astype(str)]
    return np.concatenate([np.vstack(smiles_embs), np.vstack(seq_embs)], axis=1)


def _valid_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.strip()
    cleaned["smiles"] = cleaned["smiles"].astype(str).str.strip()
    cleaned["true_value"] = pd.to_numeric(cleaned["true_value"], errors="coerce")
    cleaned["true_log10"] = pd.to_numeric(cleaned["true_log10"], errors="coerce")
    cleaned = cleaned.dropna(subset=["true_value", "true_log10"])
    cleaned = cleaned[cleaned["sequence"] != ""]
    cleaned = cleaned[cleaned["smiles"] != ""]
    cleaned = cleaned[np.isfinite(cleaned["true_log10"].to_numpy(dtype=float))]
    return cleaned.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    realkcat_root = Path(args.realkcat_root).resolve()
    unikp_root = Path(args.unikp_root).resolve()
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else (suite_dir / "unikp_cache").resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else (suite_dir / "unikp_models").resolve()
    make_output_dir(cache_root / "sequence")
    make_output_dir(cache_root / "smiles")
    make_output_dir(model_dir)

    try:
        import torch
        from sklearn.ensemble import ExtraTreesRegressor
        from transformers import T5EncoderModel, T5Tokenizer
        from transformers import modeling_utils as hf_modeling_utils
        from transformers.utils import import_utils as hf_import_utils
    except ImportError as exc:
        raise ImportError(
            "UniKP benchmarking requires torch, transformers, and scikit-learn in the current Python environment."
        ) from exc

    WordVocab, TrfmSeq2seq, split_smiles = _load_unikp_modules(unikp_root)
    seq_device = _resolve_device(args.device, torch)
    print(f"UniKP ProtT5 device: {seq_device}")

    # UniKP loads a trusted public ProtT5 checkpoint; allow the legacy .bin load path
    # even when the local torch version is older than the newest transformers guard expects.
    hf_import_utils.check_torch_load_is_safe = lambda: None
    hf_modeling_utils.check_torch_load_is_safe = lambda: None

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    seq_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(seq_device)
    seq_model.eval()

    vocab, smiles_encoder = _build_smiles_encoder(unikp_root, torch, WordVocab, TrfmSeq2seq)

    manifest_df = pd.read_csv(manifest_path)
    master_df = load_master_metadata(realkcat_root)
    requested_folds = None if not args.folds else {int(fold) for fold in args.folds}

    for fold in sorted(manifest_df["fold"].dropna().astype(int).unique()):
        if requested_folds is not None and int(fold) not in requested_folds:
            continue
        fold_sets = manifest_df[manifest_df["fold"].astype(int) == int(fold)].copy()
        fold_splits = load_fold_splits(
            realkcat_root=realkcat_root,
            parameter=args.parameter,
            embedding=args.embedding,
            fold=int(fold),
            split_names=("train", "val"),
            master_df=master_df,
        )
        train_order = ("train",) if args.exclude_val_from_train else ("train", "val")
        train_df = combine_splits(fold_splits, train_order)
        train_df = _valid_training_rows(train_df)
        if train_df.empty:
            raise ValueError(f"UniKP training data is empty for fold {fold}.")

        fold_truth_tables = [pd.read_csv(Path(path).resolve()) for path in fold_sets["truth_csv"]]
        prediction_df = pd.concat(fold_truth_tables, ignore_index=True)
        prediction_df = prediction_df.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
        prediction_df = _valid_training_rows(prediction_df.rename(columns={"true_value": "true_value", "true_log10": "true_log10"}))

        model_path = model_dir / f"fold{fold}_{args.parameter}_unikp.pkl"
        if model_path.exists() and not args.overwrite:
            with open(model_path, "rb") as handle:
                model = pickle.load(handle)
            print(f"Loaded cached UniKP fold model: {model_path}")
        else:
            all_sequences = pd.concat(
                [
                    train_df["sequence"].astype(str),
                    prediction_df["sequence"].astype(str),
                ],
                ignore_index=True,
            ).drop_duplicates()
            all_smiles = pd.concat(
                [
                    train_df["smiles"].astype(str),
                    prediction_df["smiles"].astype(str),
                ],
                ignore_index=True,
            ).drop_duplicates()

            _cache_sequence_embeddings(
                all_sequences.tolist(),
                cache_root=cache_root,
                tokenizer=tokenizer,
                model=seq_model,
                torch_module=torch,
                device=seq_device,
                batch_size=args.seq_batch_size,
            )
            _cache_smiles_embeddings(
                all_smiles.tolist(),
                cache_root=cache_root,
                split_smiles=split_smiles,
                vocab=vocab,
                smiles_encoder=smiles_encoder,
                torch_module=torch,
                batch_size=args.smiles_batch_size,
            )

            X_train = _load_feature_matrix(train_df, cache_root)
            y_train = train_df["true_log10"].to_numpy(dtype=float)
            model = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                n_jobs=args.n_jobs,
                random_state=args.seed + int(fold),
            )
            model.fit(X_train, y_train)
            with open(model_path, "wb") as handle:
                pickle.dump(model, handle)
            print(f"Trained UniKP fold model: {model_path}")

        for row in fold_sets.itertuples(index=False):
            set_dir = Path(row.truth_csv).resolve().parent
            pred_dir = make_output_dir(set_dir / "predictions")
            outpath = pred_dir / "unikp_standardized.csv"
            if outpath.exists() and not args.overwrite:
                print(f"Skipping {row.set_name}: existing UniKP predictions found")
                continue

            truth_df = pd.read_csv(Path(row.truth_csv).resolve())
            pred_input = truth_df.copy()
            pred_input["sequence"] = pred_input["sequence"].astype(str).str.strip()
            pred_input["smiles"] = pred_input["smiles"].astype(str).str.strip()
            valid_mask = pred_input["sequence"].ne("") & pred_input["smiles"].ne("")
            valid_df = pred_input.loc[valid_mask].copy().reset_index(drop=True)

            if valid_df.empty:
                pd.DataFrame(
                    columns=[
                        "pair_id",
                        "parameter",
                        "model_name",
                        "sequence",
                        "smiles",
                        "substrate_name",
                        "pred_value",
                        "pred_log10",
                        "pred_unit",
                    ]
                ).to_csv(outpath, index=False)
                print(f"Skipping {row.set_name}: no UniKP-compatible rows")
                continue

            _cache_sequence_embeddings(
                valid_df["sequence"].astype(str).drop_duplicates().tolist(),
                cache_root=cache_root,
                tokenizer=tokenizer,
                model=seq_model,
                torch_module=torch,
                device=seq_device,
                batch_size=args.seq_batch_size,
            )
            _cache_smiles_embeddings(
                valid_df["smiles"].astype(str).drop_duplicates().tolist(),
                cache_root=cache_root,
                split_smiles=split_smiles,
                vocab=vocab,
                smiles_encoder=smiles_encoder,
                torch_module=torch,
                batch_size=args.smiles_batch_size,
            )

            X_pred = _load_feature_matrix(valid_df, cache_root)
            pred_log10 = model.predict(X_pred)
            pred_value = np.power(10.0, pred_log10)
            standardized = pd.DataFrame(
                {
                    "pair_id": valid_df["pair_id"].astype(str),
                    "parameter": args.parameter,
                    "model_name": "UniKP",
                    "sequence": valid_df["sequence"].astype(str),
                    "smiles": valid_df["smiles"].astype(str),
                    "substrate_name": valid_df["substrate_name"].fillna("").astype(str),
                    "pred_value": pred_value,
                    "pred_log10": pred_log10,
                    "pred_unit": valid_df.get("true_unit", pd.Series([""] * len(valid_df))).astype(str),
                }
            )
            standardized.to_csv(outpath, index=False)
            print(f"Wrote UniKP predictions for {row.set_name}: {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
