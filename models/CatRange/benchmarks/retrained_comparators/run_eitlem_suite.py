#!/usr/bin/env python3
"""Train and run the EITLEM-Kinetics core predictor on the CatRange benchmark suite."""

import argparse
import copy
import hashlib
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import make_output_dir
from suite_partition_utils import load_fold_splits, load_master_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EITLEM-Kinetics per CatRange fold and predict the benchmark suite."
    )
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite directory with suite_manifest.csv.")
    parser.add_argument("--realkcat-root", required=True, help="Path to the CatRange repository root.")
    parser.add_argument("--eitlem-root", required=True, help="Path to the EITLEM-Kinetics repository root.")
    parser.add_argument("--parameter", default="kcat", choices=["kcat", "km"], help="Target parameter.")
    parser.add_argument("--embedding", default="esmc", choices=["esmc"], help="CatRange fold family.")
    parser.add_argument("--cache-dir", help="Feature cache directory. Default: <suite-dir>/eitlem_cache")
    parser.add_argument("--model-dir", help="Fold checkpoint directory. Default: <suite-dir>/eitlem_models")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="ESM/model device.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs per fold.")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum epochs before early stopping can trigger.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs. Set <=0 to disable.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation RMSE improvement to reset patience.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training and inference batch size.")
    parser.add_argument("--seq-batch-size", type=int, default=1, help="ESM1v embedding batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument(
        "--fold",
        dest="folds",
        action="append",
        type=int,
        help="Optional fold number to process. Repeatable. Default: process every fold in the suite.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Retrain models and overwrite predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_device(choice: str, torch_module):
    if choice == "cpu":
        return torch_module.device("cpu")
    if choice == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("EITLEM requested --device cuda, but CUDA is not available.")
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _canonical_sequence(sequence: str) -> str:
    sequence = re.sub(r"\s+", "", str(sequence).strip())
    return re.sub(r"[UZOB]", "X", sequence)


def _sha1_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _sequence_key(sequence: str) -> str:
    return _sha1_token(_canonical_sequence(sequence))


def _smiles_key(smiles: str) -> str:
    return _sha1_token(str(smiles).strip())


def _valid_rows(df: pd.DataFrame, Chem) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.strip()
    cleaned["smiles"] = cleaned["smiles"].astype(str).str.strip()
    cleaned["true_value"] = pd.to_numeric(cleaned["true_value"], errors="coerce")
    cleaned = cleaned.dropna(subset=["true_value"])
    cleaned = cleaned[cleaned["sequence"] != ""]
    cleaned = cleaned[cleaned["smiles"] != ""]
    cleaned = cleaned[cleaned["true_value"].to_numpy(dtype=float) > 0.0]
    valid_mask = cleaned["smiles"].apply(lambda value: Chem.MolFromSmiles(str(value).strip()) is not None)
    return cleaned.loc[valid_mask].reset_index(drop=True)


def _load_eitlem_modules(eitlem_root: Path):
    code_root = eitlem_root / "Code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from dataset import EitlemDataLoader, EitlemDataSet  # pylint: disable=import-error
    from eitlem_utils import Tester, Trainer  # pylint: disable=import-error
    from KCM import EitlemKcatPredictor  # pylint: disable=import-error
    from KMP import EitlemKmPredictor  # pylint: disable=import-error

    return EitlemDataLoader, EitlemDataSet, Tester, Trainer, EitlemKcatPredictor, EitlemKmPredictor


def _cache_sequence_embeddings(
    sequences,
    seq_cache_dir: Path,
    torch_module,
    esm_module,
    device,
    batch_size: int,
) -> None:
    seq_cache_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for sequence in sequences:
        sequence = _canonical_sequence(sequence)
        cache_path = seq_cache_dir / f"{_sequence_key(sequence)}.pt"
        if not cache_path.exists():
            pending.append(sequence)
    if not pending:
        return

    model, alphabet = esm_module.pretrained.esm1v_t33_650M_UR90S_1()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        batch = [(f"seq_{idx}", seq) for idx, seq in enumerate(chunk)]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch_module.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33].detach().cpu()

        for sequence, tokens_len, representation in zip(chunk, batch_lens.tolist(), token_representations):
            seq_tensor = representation[1:tokens_len - 1].clone()
            torch_module.save(seq_tensor, seq_cache_dir / f"{_sequence_key(sequence)}.pt")


def _build_pair_info(df: pd.DataFrame):
    pair_info = []
    smiles_lookup = {}
    for row in df.itertuples(index=False):
        seq_key = _sequence_key(row.sequence)
        smi_key = _smiles_key(row.smiles)
        smiles_lookup[smi_key] = str(row.smiles).strip()
        pair_info.append([seq_key, smi_key, float(row.true_value), [str(row.pair_id)]])
    return pair_info, smiles_lookup


def _build_model(parameter: str, device, KcatModel, KmModel):
    if parameter == "kcat":
        model = KcatModel(167, 512, 1280, 10, 0.5, 10)
    else:
        model = KmModel(167, 512, 1280, 10, 0.5, 10)
    return model.to(device)


def _predict_dataset(model, loader, torch_module, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch_module.no_grad():
        for data in loader:
            batch_preds = model(data.to(device))
            preds.extend(np.asarray(batch_preds.detach().cpu()).reshape(-1).tolist())
    return np.asarray(preds, dtype=float)


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)

    suite_dir = Path(args.suite_dir).resolve()
    manifest_path = suite_dir / "suite_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Suite manifest not found: {manifest_path}")

    realkcat_root = Path(args.realkcat_root).resolve()
    eitlem_root = Path(args.eitlem_root).resolve()
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else (suite_dir / "eitlem_cache").resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else (suite_dir / "eitlem_models").resolve()
    seq_cache_dir = make_output_dir(cache_root / "sequence_embeddings")
    make_output_dir(model_dir)

    try:
        import torch
        import esm
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "EITLEM benchmarking requires torch, fair-esm, and rdkit in the current Python environment."
        ) from exc

    (
        EitlemDataLoader,
        EitlemDataSet,
        Tester,
        Trainer,
        EitlemKcatPredictor,
        EitlemKmPredictor,
    ) = _load_eitlem_modules(eitlem_root)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device, torch)
    print(f"EITLEM device: {device}")

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
        train_df = _valid_rows(fold_splits["train"], Chem)
        val_df = _valid_rows(fold_splits["val"], Chem)
        if train_df.empty or val_df.empty:
            raise ValueError(f"EITLEM training or validation data is empty for fold {fold}.")

        fold_truth_tables = [pd.read_csv(Path(path).resolve()) for path in fold_sets["truth_csv"]]
        prediction_df = pd.concat(fold_truth_tables, ignore_index=True)
        prediction_df = prediction_df.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
        prediction_df = _valid_rows(prediction_df, Chem)

        all_sequences = pd.concat(
            [
                train_df["sequence"].astype(str),
                val_df["sequence"].astype(str),
                prediction_df["sequence"].astype(str),
            ],
            ignore_index=True,
        ).drop_duplicates()
        _cache_sequence_embeddings(
            all_sequences.tolist(),
            seq_cache_dir=seq_cache_dir,
            torch_module=torch,
            esm_module=esm,
            device=device,
            batch_size=args.seq_batch_size,
        )

        train_pair_info, train_smiles = _build_pair_info(train_df)
        val_pair_info, val_smiles = _build_pair_info(val_df)
        smiles_lookup = {}
        smiles_lookup.update(train_smiles)
        smiles_lookup.update(val_smiles)
        if not prediction_df.empty:
            _, pred_smiles = _build_pair_info(prediction_df)
            smiles_lookup.update(pred_smiles)

        model_path = model_dir / f"fold{fold}_{args.parameter}_eitlem_core.pt"
        if model_path.exists() and not args.overwrite:
            model = _build_model(args.parameter, device, EitlemKcatPredictor, EitlemKmPredictor)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded cached EITLEM fold model: {model_path}")
        else:
            train_set = EitlemDataSet(
                train_pair_info,
                str(seq_cache_dir),
                smiles_lookup,
                1024,
                4,
                True,
                "MACCSKeys",
            )
            val_set = EitlemDataSet(
                val_pair_info,
                str(seq_cache_dir),
                smiles_lookup,
                1024,
                4,
                True,
                "MACCSKeys",
            )
            train_loader = EitlemDataLoader(
                data=train_set,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                pin_memory=device.type == "cuda",
            )
            val_loader = EitlemDataLoader(
                data=val_set,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                pin_memory=device.type == "cuda",
            )

            model = _build_model(args.parameter, device, EitlemKcatPredictor, EitlemKmPredictor)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[max(1, args.epochs // 2), max(1, int(args.epochs * 0.8))],
                gamma=0.9,
            )
            loss_fn = torch.nn.MSELoss()
            trainer = Trainer(device, loss_fn, log10=True)
            tester = Tester(device, loss_fn, log10=True)
            best_rmse = None
            best_state = None
            best_epoch = None
            stale_epochs = 0

            for epoch in range(1, args.epochs + 1):
                train_mae, train_rmse, train_r2, train_loss = trainer.run(
                    model,
                    train_loader,
                    optimizer,
                    len(train_pair_info),
                    f"fold{fold} epoch {epoch} train",
                )
                val_mae, val_rmse, val_r2, val_loss = tester.test(
                    model,
                    val_loader,
                    len(val_pair_info),
                    f"fold{fold} epoch {epoch} val",
                )
                scheduler.step()
                print(
                    f"fold {fold} epoch {epoch}/{args.epochs}: "
                    f"train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f} "
                    f"train_mae={train_mae:.4f} val_mae={val_mae:.4f} "
                    f"train_r2={train_r2:.4f} val_r2={val_r2:.4f} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )
                if best_rmse is None or val_rmse < (best_rmse - args.min_delta):
                    best_rmse = val_rmse
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                elif args.patience > 0 and epoch >= args.min_epochs:
                    stale_epochs += 1
                    if stale_epochs >= args.patience:
                        print(
                            f"fold {fold}: early stopping at epoch {epoch} "
                            f"(best_epoch={best_epoch}, best_val_rmse={best_rmse:.4f})"
                        )
                        break

            if best_state is None:
                raise RuntimeError(f"EITLEM training failed to produce a checkpoint for fold {fold}.")
            torch.save(best_state, model_path)
            model.load_state_dict(best_state)
            print(
                f"Trained EITLEM fold model: {model_path} "
                f"(best_epoch={best_epoch}, best_val_rmse={best_rmse:.4f})"
            )

        for row in fold_sets.itertuples(index=False):
            set_dir = Path(row.truth_csv).resolve().parent
            pred_dir = make_output_dir(set_dir / "predictions")
            outpath = pred_dir / "eitlem_standardized.csv"
            if outpath.exists() and not args.overwrite:
                print(f"Skipping {row.set_name}: existing EITLEM predictions found")
                continue

            truth_df = pd.read_csv(Path(row.truth_csv).resolve())
            valid_df = _valid_rows(truth_df, Chem)
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
                print(f"Skipping {row.set_name}: no EITLEM-compatible rows")
                continue

            pair_info, pred_smiles_lookup = _build_pair_info(valid_df)
            smiles_lookup.update(pred_smiles_lookup)
            pred_set = EitlemDataSet(
                pair_info,
                str(seq_cache_dir),
                smiles_lookup,
                1024,
                4,
                True,
                "MACCSKeys",
            )
            pred_loader = EitlemDataLoader(
                data=pred_set,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                pin_memory=device.type == "cuda",
            )
            pred_log10 = _predict_dataset(model, pred_loader, torch, device)
            pred_value = np.power(10.0, pred_log10)
            standardized = pd.DataFrame(
                {
                    "pair_id": valid_df["pair_id"].astype(str),
                    "parameter": args.parameter,
                    "model_name": "EITLEM-Kinetics",
                    "sequence": valid_df["sequence"].astype(str),
                    "smiles": valid_df["smiles"].astype(str),
                    "substrate_name": valid_df["substrate_name"].fillna("").astype(str),
                    "pred_value": pred_value,
                    "pred_log10": pred_log10,
                    "pred_unit": valid_df.get("true_unit", pd.Series([""] * len(valid_df))).astype(str),
                }
            )
            standardized.to_csv(outpath, index=False)
            print(f"Wrote EITLEM predictions for {row.set_name}: {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
