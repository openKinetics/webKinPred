#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe-enzyme evaluation on the DLKcat dataset
─────────────────────────────────────────────
• Pick three (EC, Organism, Smiles) triplets (≤ 15 total rows each,,
  skipping the hard-coded EC ‘1.1.1.9’).
• For every probe enzyme:
      – find its wild-type (WT) sequence,
      – cluster *all* DLKcat sequences at 90 % identity,
      – remove every datapoint whose sequence falls into the WT’s cluster
        (thus, no homologs ≥ 0.90 identical are seen during training),
      – train two models (KinForm-L ≡ CONFIG_L and UniKP ≡ CONFIG_UniKP),
      – predict kcat for the WT and every mutant.
• Collect real kcat and both predictions in one tidy, wide dataframe
  (row = enzyme cluster, columns = WT + mutants + “_<model>” predictions)
  and save to CSV.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add parent dir to path

# ───────────────────── repo-provided helpers & configs ───────────────────── #
from config import (  # type: ignore
    RAW_DLKCAT,
    SEQ_LOOKUP,
    BS_PRED_PATH,
    CONFIG_L,
    CONFIG_H,
    CONFIG_UniKP,
    ROOT
)
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab  
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks 
from utils.pca import make_design_matrices  
from model_training import train_model  
from config import MMSEQS_ENV, CONDA_EXE  # type: ignore
# ───────────────────── oversampling helpers ───────────────────── #
from utils.oversampling import (
    oversample_similarity_balanced_indices,
    oversample_kcat_balanced_indices,
)

# ─────────────────── mmseqs similarity search helper ──────────────────── #
from typing import Sequence, Set
import tempfile, subprocess

def get_similar_indices(
    sequences: Sequence[str],
    test_idx: Sequence[int],
    *,
    min_id: float = 0.20,
    cov: float = 0.80,
) -> Set[int]:
    """
    Return the set of row indices whose sequence is ≥ `min_id`
    identical to *any* sequence in `test_idx` (self-hits included).

    Implemented with `mmseqs easy-search` so we do *not* rely on
    clustering heuristics – every pairwise hit is examined exactly once.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ── write full corpus (targets) ──
        target_fa = td / "targets.fa"
        with target_fa.open("w") as fh:
            for i, seq in enumerate(sequences):
                fh.write(f">{i}\n{seq}\n")

        # ── write WT + mutant queries ──
        query_fa = td / "query.fa"
        with query_fa.open("w") as fh:
            for i in test_idx:
                fh.write(f">{i}\n{sequences[i]}\n")

        out_tsv = td / "hits.tsv"

        subprocess.run(
            [
                str(CONDA_EXE),
                "run",
                "-n",
                MMSEQS_ENV,
                "mmseqs",
                "easy-search",
                str(query_fa),
                str(target_fa),
                str(out_tsv),
                str(td),
                "--min-seq-id",
                str(min_id),
                "--cov-mode",
                "0",
                "-c",
                str(cov),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # ── parse hits:  col-0 = query-ID, col-1 = target-ID ──
        hits: Set[int] = set()
        with out_tsv.open() as fh:
            for line in fh:
                _, target_id, *_ = line.rstrip().split("\t")
                hits.add(int(target_id))

    return hits

# ─────────────────────────── convenience utilities ───────────────────────── #
def infer_mutations(wt: str, mut: str) -> str:
    """Return comma-separated mutation codes (e.g. ‘E124A’), or ‘wildtype’."""
    if len(wt) != len(mut):
        raise ValueError("WT and mutant sequences differ in length")
    muts = [f"{wt[i]}{i+1}{mut[i]}" for i in range(len(wt)) if wt[i] != mut[i]]
    return "wildtype" if not muts else ",".join(muts)

def _pick_ten_indices(group_df: pd.DataFrame) -> List[int]:
    """
    Return up to 10 row indices that cover the widest possible kcat range:
    • wild-type (mandatory, exactly one) plus nine evenly-spaced mutants.
    """
    wt_row = group_df[group_df["Type"] == "wildtype"]
    if len(wt_row) != 1:
        return []                       # invalid group (0 or >1 WT)
    wt_idx = int(wt_row.index[0])

    # Remaining rows sorted by kcat
    rest = group_df.drop(wt_idx).sort_values("Value")
    if len(rest) < 9:
        return []                       # not enough mutants to reach 10 points

    # Pick nine evenly-spaced indices across the sorted mutants
    step_positions = np.linspace(0, len(rest) - 1, 9, dtype=int)
    mutant_idx = rest.index[step_positions].tolist()

    return [wt_idx] + mutant_idx        # WT first, then nine mutants

def choose_probe_triplets(
    df: pd.DataFrame,
    k: int = 3,
    n_candidates: int = 100,
) -> Tuple[List[Tuple[str, str, str]], Dict[Tuple[str, str, str], List[int]]]:
    """
    • Examine the `n_candidates` largest (EC, Organism, Smiles) groups.
    • For each group, try to pick 10 indices using `_pick_ten_indices`.
    • Score the group by the kcat range of those ten points.
    • Return:
          1. a list of the best `k` (EC, Org, Smiles) triplets
          2. a dict  triplet → list[10 indices]   (to use as test_idx later).
    """

    # Rank groups by size (largest first) and keep the first `n_candidates`
    group_sizes = (
        df.groupby(["ECNumber", "Organism", "Smiles"])
          .size()
          .sort_values(ascending=False)
    ).head(n_candidates)

    candidates: List[Tuple[float, Tuple[str, str, str], List[int]]] = []
    for (ec, org, smi), _ in group_sizes.iteritems():
        gdf = df[(df["ECNumber"] == ec) & (df["Organism"] == org) & (df["Smiles"] == smi)]

        wt_row = gdf[gdf["Type"] == "wildtype"]
        if len(wt_row) != 1: # invalid group (0 or >1 WT)
            continue
        if float(wt_row["Value"].values[0]) > 100: # skip groups with WT kcat > 100
            continue
        chosen = _pick_ten_indices(gdf)
        if not chosen:
            continue
        kcat_wt = float(gdf.loc[chosen[0], "Value"])
        log_wt = np.log10(kcat_wt)
        mutant_vals = gdf.loc[chosen[1:], "Value"].astype(float)
        log_mutants = np.log10(mutant_vals)

        if any(abs(log_wt - lm) > 2 for lm in log_mutants):
            continue
        wt_seq = gdf.loc[chosen[0], "Sequence"]  # first is always WT
        mutant_seqs = gdf.loc[chosen[1:], "Sequence"]  # exclude WT

        def mutation_count(mut_seq: str) -> int:
            return sum(1 for a, b in zip(wt_seq, mut_seq) if a != b)

        if any(mutation_count(seq) > 1 for seq in mutant_seqs):
            continue  # skip group

        vals = df.loc[chosen, "Value"].astype(float)
        spread = vals.max() - vals.min()
        candidates.append((spread, (ec, org, smi), chosen))

    # Sort by spread, ensure each EC appears at most once
    candidates.sort(reverse=True, key=lambda x: x[0])

    hardcoded_triplets = [
        ("1.14.99.1", "Ovis aries", "OO"),
        ("1.5.1.3", "Escherichia coli", "C1C(=NC2=C(N1)N=C(NC2=O)N)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)[O-])C(=O)[O-]"),
        ("2.7.2.4", "Thermus thermophilus", "C(C(C(=O)O)N)C(=O)O"),
        ("6.1.1.15", "Escherichia coli", "C1CC(NC1)C(=O)O"),
    ]

    probes: List[Tuple[str, str, str]] = []
    probe_indices: Dict[Tuple[str, str, str], List[int]] = {}
    seen_ec: set[str] = set()

    # Add hardcoded first
    for triplet in hardcoded_triplets:
        ec, org, smi = triplet
        gdf = df[(df["ECNumber"] == ec) & (df["Organism"] == org) & (df["Smiles"] == smi)]

        wt_row = gdf[gdf["Type"] == "wildtype"]
        if len(wt_row) != 1:
            continue
        if float(wt_row["Value"].values[0]) > 100:
            continue
        chosen = _pick_ten_indices(gdf)
        if not chosen:
            continue
        kcat_wt = float(gdf.loc[chosen[0], "Value"])
        log_wt = np.log10(kcat_wt)
        mutant_vals = gdf.loc[chosen[1:], "Value"].astype(float)
        log_mutants = np.log10(mutant_vals)
        if any(abs(log_wt - lm) > 3 for lm in log_mutants):
            continue
        wt_seq = gdf.loc[chosen[0], "Sequence"]
        mutant_seqs = gdf.loc[chosen[1:], "Sequence"]
        if any(sum(1 for a, b in zip(wt_seq, s) if a != b) > 2 for s in mutant_seqs):
            continue

        probes.append(triplet)
        probe_indices[triplet] = chosen
        seen_ec.add(ec)

    # Add remaining to reach k total
    for spread, triplet, idx_list in candidates:
        ec = triplet[0]
        if ec in seen_ec:
            continue
        probes.append(triplet)
        probe_indices[triplet] = idx_list
        seen_ec.add(ec)
        if len(probes) == k:
            break

    return probes, probe_indices

def assemble_row(
    enz_df: pd.DataFrame,
    preds: Dict[str, np.ndarray],
) -> pd.Series:
    """Build one wide row: real kcats first, then predictions per model."""
    wt_seq = enz_df.loc[enz_df["Type"] == "wildtype", "Sequence"].iloc[0]
    enz_df = enz_df.assign(
        Mutation=enz_df["Sequence"].apply(lambda s: infer_mutations(wt_seq, s))
    ).sort_values("Mutation")

    data: Dict[str, float] = {
        mut: float(kcat) for mut, kcat in zip(enz_df["Mutation"], enz_df["Value"])
    }
    for model, yhat in preds.items():
        for mut, pred in zip(enz_df["Mutation"], yhat):
            data[f"{mut}_{model}"] = float(pred)

    key = (enz_df["ECNumber"].iloc[0], enz_df["Organism"].iloc[0], enz_df["Smiles"].iloc[0])
    return pd.Series(data, name=key)


# ────────────────────────────────── main ──────────────────────────────────── #
def main() -> None:
    # 1) ─────────── load DLKcat raw JSON (basic filtering) ──────────────── #
    raw = [
        d
        for d in json.loads(Path(RAW_DLKCAT).read_text())
        if len(d["Sequence"]) <= 1_499
        and float(d["Value"]) > 0
        and "." not in d["Smiles"]
    ]
    df = pd.DataFrame(raw)

    # 2) ───────────────────── select three probe enzymes ─────────────────── #
    probes, probe_indices = choose_probe_triplets(df, k=9)
    print("\nChosen probe enzymes:")
    print("hardcoded_triplets = [")
    for ec, org, smi in probes:
        print(f'    ("{ec}", "{org}", "{smi}"),')
    print("]")

    # 3) ─────────────── global resources (shared across configs) ─────────── #
    sequences = df["Sequence"].tolist()
    smiles_all = df["Smiles"].tolist()
    y_log = np.log10(df["Value"].astype(float).to_numpy(np.float32))

    seq_to_id = {v: k for k, v in pd.read_pickle(SEQ_LOOKUP).items()}

    bs_df = pd.read_csv(BS_PRED_PATH, sep="\t")
    smiles_vec = smiles_to_vec(smiles_all, method="smiles_transformer")
    # ─────────────── estimate number of sequences to drop ─────────────── #
    mid_id = 0.10
    print(f"\nExpected rows to drop per probe enzyme (≥ {mid_id} identical):") 
    for ec, org, smi in probes:
        key        = (ec, org, smi)
        test_idx   = np.array(probe_indices[key], dtype=int)
        drop_set   = get_similar_indices(sequences, test_idx, min_id=mid_id, cov=0.10)
        print(f"  • {org} | {ec} | {smi} → {len(drop_set)} rows to drop")


    # ───────────────────────────── 5. fit two configs ───────────────────────── #
    configs = [CONFIG_L, CONFIG_UniKP]
    pred_store: Dict[str, Dict[Tuple[str, str, str], np.ndarray]] = {
        c["name"]: {} for c in configs
    }

    for cfg in configs:
        name = cfg["name"]
        print(f"\n==== Starting config: {name} ====")

        # Extract feature blocks once per config
        blocks, block_names = sequences_to_feature_blocks(
            sequences,
            bs_df,
            seq_to_id,
            use_esmc=cfg["use_esmc"],
            use_esm2=cfg["use_esm2"],
            use_t5=cfg["use_t5"],
            t5_last_layer=cfg["t5_last_layer"],
            prot_rep_mode=cfg["prot_rep_mode"],
            task="kcat",
        )

        for ec, org, smi in probes:
            # Re‐define mask so that we can use it to find WT
            mask = (
                (df["ECNumber"] == ec)
                & (df["Organism"] == org)
                & (df["Smiles"] == smi)
            )
            test_idx = np.array(probe_indices[(ec, org, smi)], dtype=int)

            # Find wildtype row via mask
            wt_rows = df.loc[mask & (df["Type"] == "wildtype")]
            if wt_rows.empty:
                print(f"⚠️  Skipping {org} ({ec}) — no wildtype found.")
                continue

            similar_idx = np.fromiter(
                get_similar_indices(sequences, test_idx, min_id=mid_id, cov=0.80),
                dtype=int,
            )
            to_remove = np.union1d(similar_idx, test_idx)
            train_idx = np.setdiff1d(np.arange(len(df)), to_remove)

            print(
                f"→ Training on enzyme: {org} ({ec})\n"
                f"   • Cluster size removed: {len(to_remove)}\n"
                f"   • Test set size      : {len(test_idx)}\n"
                f"   • Train set size     : {len(train_idx)}"
            )

            # Build design matrices
            X_tr, X_te,_ = make_design_matrices(
                train_idx, test_idx, blocks, block_names, cfg, smiles_vec
            )
            y_tr = y_log[train_idx]
            y_te = y_log[test_idx]

            key = (ec, org, smi)

            if name.startswith("KinForm-L"):
                # ─────────── OVERSAMPLED KinForm-L ───────────
                tr_bal = oversample_similarity_balanced_indices(train_idx, sequences)
                print(f"      Oversampling-1 (similarity): {len(tr_bal)} rows from {len(train_idx)}")

                tr_bal = oversample_kcat_balanced_indices(tr_bal, y_log)
                print(f"      Oversampling-2 (kcat)      : {len(tr_bal)} rows after second step")

                X_tr_os, X_te,_ = make_design_matrices(
                    tr_bal, test_idx, blocks, block_names, cfg, smiles_vec
                )
                y_tr_os = y_log[tr_bal]

                _, yhat_log, _ = train_model(
                    X_tr_os, y_tr_os, X_te, y_te, fold=42, n_jobs=-1
                )
            else:
                # ─────────── baseline training (UniKP etc.) ───────────
                X_tr, X_te,_ = make_design_matrices(
                    train_idx, test_idx, blocks, block_names, cfg, smiles_vec
                )
                y_tr = y_log[train_idx]

                _, yhat_log, _ = train_model(
                    X_tr, y_tr, X_te, y_te, fold=42, n_jobs=-1
                )

            yhat_lin = np.power(10.0, yhat_log)
            pred_store[name][key] = yhat_lin
            print(f"   ✓ Finished {name} for {org} ({ec})")


    # 6) ───────────── assemble wide dataframe & save to CSV ──────────────── #
    long_rows = []
    for ec, org, smi in probes:
        key = (ec, org, smi)
        # Skip if any model failed to produce predictions
        if not all(key in pred_store[cfg["name"]] for cfg in configs):
            print(f"⚠️  Skipping {org} ({ec}) during result assembly — missing predictions.")
            continue

        # Get the ten test indices for this group (WT + 9 mutants)
        test_idx = probe_indices[key]  # list of length 10

        # Fetch predictions (each is length-10 array) for this key
        preds_per_model = {cfg["name"]: pred_store[cfg["name"]][key] for cfg in configs}

        # For each of the 10 indices, extract real kcat + mutation, then each model’s prediction
        for pos, row_idx in enumerate(test_idx):
            row = df.loc[row_idx]
            mutation = infer_mutations(
                df.loc[test_idx[0], "Sequence"],  # wild-type sequence is at position 0
                row["Sequence"]
            )
            kcat_real = float(row["Value"])

            for model in configs:
                model_name = model["name"]
                # same position pos in the yhat_lin array
                kcat_pred = float(preds_per_model[model_name][pos])

                long_rows.append({
                    "ECNumber": ec,
                    "Organism": org,
                    "Smiles": smi,
                    "Mutation": mutation,
                    "kcat_real": kcat_real,
                    "kcat_pred": kcat_pred,
                    "Model": model_name
                })

    long_df = pd.DataFrame(long_rows)
    out_long = ROOT / "results/probe_enzyme_predictions_long.csv"
    long_df.to_csv(out_long, index=False)
    print("\n Long-format results written to:", out_long)


if __name__ == "__main__":
    main()
