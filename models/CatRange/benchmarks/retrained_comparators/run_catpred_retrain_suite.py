#!/usr/bin/env python3
"""Retrain CatPred per CatRange fold and predict every suite set for that fold."""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from benchmark_utils import canonical_protein_mask, make_output_dir, stable_sequence_ids, write_json
from run_catpred_suite import _prepare_catpred_subset, _standardize
from suite_partition_utils import load_fold_splits, load_master_metadata


ESM_MAX_LENGTH = 2048


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain CatPred on CatRange folds and predict every suite set."
    )
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument("--realkcat-root", required=True, help="Path to the CatRange repository root.")
    parser.add_argument("--repo-root", required=True, help="Path to the CatPred repository root.")
    parser.add_argument("--python-executable", required=True, help="Python executable for the CatPred env.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km", "ki"], help="CatPred parameter.")
    parser.add_argument("--embedding", default="esmc", choices=["esmc"], help="CatRange fold family.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="CatPred/ESM device.")
    parser.add_argument("--gpu-index", type=int, default=0, help="CUDA device index for CatPred when device=cuda.")
    parser.add_argument("--cache-dir", help="Optional ESM2 cache directory. Default: <suite-dir>/catpred_esm2_cache")
    parser.add_argument("--model-dir", help="Fold checkpoint root. Default: <suite-dir>/catpred_retrained")
    parser.add_argument(
        "--esm-vendor",
        help="Optional path added to PYTHONPATH so the CatPred env can import the vendored esm package.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold.")
    parser.add_argument("--ensemble-size", type=int, default=1, help="CatPred ensemble size per fold.")
    parser.add_argument("--batch-size", type=int, default=16, help="CatPred batch size.")
    parser.add_argument("--seq-embed-dim", type=int, default=36, help="CatPred sequence projection width.")
    parser.add_argument("--seq-self-attn-nheads", type=int, default=6, help="CatPred sequence attention heads.")
    parser.add_argument(
        "--loss-function",
        default="mve",
        choices=["mse", "mve", "evidential", "dirichlet"],
        help="CatPred regression loss function.",
    )
    parser.add_argument("--max-batch-size", type=int, default=4, help="Maximum sequences per ESM2 batch.")
    parser.add_argument("--max-batch-tokens", type=int, default=6000, help="Maximum padded tokens per ESM2 batch.")
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
        raise RuntimeError("CatPred requested device=cuda, but CUDA is not available in this environment.")
    return torch.device(raw_device)


def _batch_records(records, max_batch_size: int, max_batch_tokens: int):
    sorted_records = sorted(records, key=lambda row: len(row["sequence"]))
    batch = []
    padded_len = 0

    for record in sorted_records:
        seq_tokens = min(len(record["sequence"]) + 2, ESM_MAX_LENGTH)
        next_padded_len = max(padded_len, seq_tokens)
        next_batch_size = len(batch) + 1
        next_cost = next_padded_len * next_batch_size

        if batch and (next_batch_size > max_batch_size or next_cost > max_batch_tokens):
            yield batch
            batch = []
            padded_len = 0
            next_padded_len = seq_tokens

        batch.append(record)
        padded_len = next_padded_len

    if batch:
        yield batch


def _ensure_esm_import(esm_vendor):
    if esm_vendor:
        vendor_path = str(Path(esm_vendor).resolve())
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
    try:
        import esm  # pylint: disable=import-error
    except ImportError as exc:
        raise ImportError(
            "CatPred retraining requires the esm package. Pass --esm-vendor if it lives outside the env."
        ) from exc
    return esm


def _compute_and_store_batch(batch, model, batch_converter, device: torch.device, cache_dir: Path) -> None:
    data = [(row["sequence_id"], row["sequence"]) for row in batch]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens[:, :ESM_MAX_LENGTH]
    if device.type == "cuda":
        batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_representations = results["representations"][33]
    for idx, row in enumerate(batch):
        out_path = cache_dir / f"{row['sequence_id']}.pt"
        seq_len = len(row["sequence"])
        embedding = token_representations[idx][1 : seq_len + 1].detach().cpu()
        torch.save(embedding, out_path)


def _ensure_cached_esm2(records, cache_dir: Path, esm_vendor, device: torch.device, max_batch_size: int, max_batch_tokens: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [record for record in records if not (cache_dir / f"{record['sequence_id']}.pt").exists()]
    if not pending:
        return

    esm = _ensure_esm_import(esm_vendor)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if device.type == "cuda":
        model = model.to(device)

    processed = 0
    for batch in _batch_records(pending, max_batch_size=max_batch_size, max_batch_tokens=max_batch_tokens):
        _compute_and_store_batch(batch, model, batch_converter, device=device, cache_dir=cache_dir)
        processed += len(batch)
        print(f"Cached CatPred ESM2 embeddings: {processed}/{len(pending)}")


def _catpred_target_column(parameter: str) -> str:
    return {
        "kcat": "log10kcat",
        "km": "log10km",
        "ki": "log10ki",
    }[parameter]


def _clean_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.strip()
    cleaned["smiles"] = cleaned["smiles"].astype(str).str.strip()
    cleaned["true_log10"] = pd.to_numeric(cleaned["true_log10"], errors="coerce")
    substrate_series = cleaned["substrate_name"] if "substrate_name" in cleaned.columns else pd.Series("", index=cleaned.index)
    cleaned["substrate_name"] = substrate_series.fillna("").astype(str)
    cleaned = cleaned[canonical_protein_mask(cleaned["sequence"])]
    cleaned = cleaned[cleaned["sequence"] != ""]
    cleaned = cleaned[cleaned["smiles"] != ""]
    cleaned = cleaned[cleaned["true_log10"].notna()]
    cleaned = cleaned.reset_index(drop=True)
    cleaned["pdbpath"] = (
        cleaned["sequence_id"].astype(str)
        if "sequence_id" in cleaned.columns
        else stable_sequence_ids(cleaned["sequence"])
    )
    return cleaned


def _write_fold_csv(df: pd.DataFrame, outpath: Path, parameter: str) -> Path:
    target_col = _catpred_target_column(parameter)
    catpred_df = pd.DataFrame(
        {
            "pair_id": df["pair_id"].astype(str),
            "SMILES": df["smiles"].astype(str),
            "sequence": df["sequence"].astype(str),
            "pdbpath": df["pdbpath"].astype(str),
            "substrate_name": df["substrate_name"].astype(str),
            target_col: df["true_log10"].astype(float),
        }
    )
    catpred_df.to_csv(outpath, index=False)
    return outpath


def _write_records_json(df: pd.DataFrame, cache_dir: Path, outpath: Path) -> Path:
    records = {}
    subset = df[["pdbpath", "sequence"]].drop_duplicates().sort_values("pdbpath")
    for row in subset.itertuples(index=False):
        cache_file = cache_dir / f"{row.pdbpath}.pt"
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing cached ESM2 tensor: {cache_file}")
        records[str(row.pdbpath)] = {
            "name": str(row.pdbpath),
            "seq": str(row.sequence),
            "esm2_feats_path": str(cache_file.resolve()),
        }

    with gzip.open(outpath, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)
    return outpath


def _training_env(esm_vendor):
    env = os.environ.copy()
    if esm_vendor:
        vendor_path = str(Path(esm_vendor).resolve())
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = vendor_path if not current else f"{vendor_path}:{current}"
    return env


def _run_subprocess(cmd, cwd=None, env=None):
    rendered = " ".join(shlex_quote(part) for part in cmd)
    print(f"$ {rendered}")
    subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, env=env, check=True)


def shlex_quote(value) -> str:
    import shlex
    return shlex.quote(str(value))


def _train_fold(args, fold: int, train_csv: Path, val_csv: Path, test_csv: Path, records_json: Path, checkpoint_dir: Path):
    cmd = [
        Path(args.python_executable).resolve(),
        Path(args.repo_root).resolve() / "train.py",
        "--protein_records_path",
        records_json,
        "--data_path",
        train_csv,
        "--dataset_type",
        "regression",
        "--separate_val_path",
        val_csv,
        "--separate_test_path",
        test_csv,
        "--smiles_columns",
        "SMILES",
        "--target_columns",
        _catpred_target_column(args.parameter),
        "--extra_metrics",
        "mae",
        "mse",
        "r2",
        "--ensemble_size",
        str(args.ensemble_size),
        "--seq_embed_dim",
        str(args.seq_embed_dim),
        "--seq_self_attn_nheads",
        str(args.seq_self_attn_nheads),
        "--loss_function",
        args.loss_function,
        "--batch_size",
        str(args.batch_size),
        "--save_dir",
        checkpoint_dir,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed + int(fold)),
        "--pytorch_seed",
        str(args.seed + int(fold)),
        "--add_esm_feats",
        "--quiet",
    ]

    resolved_device = _resolve_device(args.device)
    if resolved_device.type == "cuda":
        cmd.extend(["--gpu", str(args.gpu_index)])
    else:
        cmd.append("--no_cuda")

    _run_subprocess(cmd, cwd=Path(args.repo_root).resolve(), env=_training_env(args.esm_vendor))


def main() -> int:
    args = parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    repo_root = Path(args.repo_root).resolve()
    realkcat_root = Path(args.realkcat_root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (suite_dir / "catpred_esm2_cache").resolve()
    model_root = Path(args.model_dir).resolve() if args.model_dir else (suite_dir / "catpred_retrained").resolve()
    make_output_dir(model_root)

    sys.path.insert(0, str(repo_root))
    from catpred.inference import PredictionRequest, run_prediction_pipeline  # pylint: disable=import-error

    manifest_df = pd.read_csv(manifest_path)
    master_df = load_master_metadata(realkcat_root)
    esm_device = _resolve_device(args.device)
    requested_folds = None if not args.folds else {int(fold) for fold in args.folds}

    for fold in sorted(manifest_df["fold"].dropna().astype(int).unique()):
        if requested_folds is not None and int(fold) not in requested_folds:
            continue
        print(f"=== CatPred fold {int(fold)} ===")
        fold_sets = manifest_df[manifest_df["fold"].astype(int) == int(fold)].copy()
        split_frames = load_fold_splits(
            realkcat_root=realkcat_root,
            parameter=args.parameter,
            embedding=args.embedding,
            fold=int(fold),
            split_names=("train", "val", "test"),
            master_df=master_df,
        )
        train_df = _clean_training_rows(split_frames["train"])
        val_df = _clean_training_rows(split_frames["val"])
        test_df = _clean_training_rows(split_frames["test"])
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(f"CatPred fold {fold} has an empty train/val/test split after cleaning.")

        fold_dir = make_output_dir(model_root / f"fold{fold}")
        data_dir = make_output_dir(fold_dir / "data")
        checkpoint_dir = make_output_dir(fold_dir / "checkpoints")
        records_json = fold_dir / "protein_records.json.gz"

        train_csv = _write_fold_csv(train_df, data_dir / "train.csv", args.parameter)
        val_csv = _write_fold_csv(val_df, data_dir / "val.csv", args.parameter)
        test_csv = _write_fold_csv(test_df, data_dir / "test.csv", args.parameter)

        union_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        union_records = [
            {"sequence_id": str(row.pdbpath), "sequence": str(row.sequence)}
            for row in union_df[["pdbpath", "sequence"]].drop_duplicates().itertuples(index=False)
        ]
        print(
            f"Preparing CatPred fold {int(fold)} cache: "
            f"{len(union_records)} unique sequences across train/val/test"
        )
        _ensure_cached_esm2(
            union_records,
            cache_dir=cache_dir,
            esm_vendor=args.esm_vendor,
            device=esm_device,
            max_batch_size=args.max_batch_size,
            max_batch_tokens=args.max_batch_tokens,
        )
        _write_records_json(union_df, cache_dir=cache_dir, outpath=records_json)

        metadata = {
            "fold": int(fold),
            "parameter": args.parameter,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "records_json": str(records_json.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
        }
        write_json(fold_dir / "fold_metadata.json", metadata)

        marker = checkpoint_dir / "fold_training_complete.json"
        if args.overwrite or not marker.exists():
            if args.overwrite and checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _train_fold(
                args=args,
                fold=int(fold),
                train_csv=train_csv,
                val_csv=val_csv,
                test_csv=test_csv,
                records_json=records_json,
                checkpoint_dir=checkpoint_dir,
            )
            write_json(
                marker,
                {
                    "fold": int(fold),
                    "checkpoint_dir": str(checkpoint_dir.resolve()),
                    "parameter": args.parameter,
                },
            )
        else:
            print(f"Skipping CatPred training for fold {fold}: existing checkpoint marker found")

        for row in fold_sets.itertuples(index=False):
            print(f"Running CatPred prediction for {row.set_name} with fold {int(fold)} checkpoint")
            set_dir = Path(row.truth_csv).resolve().parent
            pred_dir = make_output_dir(set_dir / "predictions")
            raw_dir = make_output_dir(set_dir / "catpred_raw")
            standardized_output = pred_dir / "catpred_standardized.csv"
            raw_output = raw_dir / "catpred_output.csv"
            if standardized_output.exists() and not args.overwrite:
                print(f"Skipping CatPred prediction for {row.set_name}: existing standardized output found")
                continue

            valid_df, filtered_input_csv, filtered_records_json = _prepare_catpred_subset(
                Path(row.catpred_input_csv).resolve(),
                records_json,
                raw_dir,
                pred_dir,
            )
            if valid_df.empty:
                pd.DataFrame(
                    columns=[
                        "pair_id",
                        "parameter",
                        "model_name",
                        "sequence",
                        "smiles",
                        "substrate_name",
                        "pdbpath",
                        "pred_value",
                        "pred_log10",
                        "pred_unit",
                        "sd_total",
                        "sd_aleatoric",
                        "sd_epistemic",
                    ]
                ).to_csv(standardized_output, index=False)
                print(f"Skipping {row.set_name}: no CatPred-compatible sequences in this set")
                continue

            request = PredictionRequest(
                parameter=args.parameter,
                input_file=str(filtered_input_csv),
                checkpoint_dir=str(checkpoint_dir.resolve()),
                use_gpu=esm_device.type == "cuda",
                repo_root=str(repo_root),
                python_executable=str(Path(args.python_executable).resolve()),
                protein_records_file=str(filtered_records_json),
            )
            produced_output = Path(run_prediction_pipeline(request, results_dir=str(raw_dir)))
            if not produced_output.exists():
                raise FileNotFoundError(f"CatPred did not create the expected output: {produced_output}")

            raw_df = pd.read_csv(produced_output)
            raw_df.to_csv(raw_output, index=False)
            standardized_df = _standardize(raw_df, args.parameter)
            standardized_df.to_csv(standardized_output, index=False)
            print(f"Wrote retrained CatPred predictions for {row.set_name}: {standardized_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
