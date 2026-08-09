#!/usr/bin/env python3
"""CatRange-only inference from protein sequence and substrate SMILES.

This module is scoped to the CatRange manuscript model: ESM-C protein
embeddings, ChemBERTa substrate embeddings, and XGBoost kinetic-regime
classification for kcat and KM. It does not include CatRange-Regressor,
CatRange-Lens, or TokenLens.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - graceful fallback for missing torch runtime
    torch = None


BIN_EDGES = {
    "kcat": np.asarray([0, 1e-8, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e8], dtype=float),
    "km": np.asarray([1e-14, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e4], dtype=float),
}


def _require_runtime() -> None:
    if torch is None:
        raise RuntimeError("CatRange requires torch to be installed in the runtime environment")


def _log_bin_centers(parameter: str) -> np.ndarray:
    edges = BIN_EDGES[parameter].copy()
    safe = edges.copy()
    if safe[0] <= 0:
        safe[0] = safe[1]
    logs = np.log10(safe)
    if edges[0] <= 0:
        logs[0] = logs[1]
    centers = (logs[:-1] + logs[1:]) / 2.0
    centers[0] = logs[1]
    return centers.astype(np.float32)


def _bin_labels(parameter: str) -> list[str]:
    edges = BIN_EDGES[parameter]
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        if parameter == "kcat":
            labels.append(f"{low:g} to {high:g} s^-1")
        else:
            labels.append(f"{low:g} to {high:g} M")
    return labels


def _candidate_model_names(parameter: str) -> list[str]:
    parameter = parameter.lower()
    return [
        f"{parameter}_model_v1b.pkl",
        f"{parameter}_esmc_FINAL.pkl",
        f"{parameter}_model.pkl",
        f"{parameter}.pkl",
    ]


def _candidate_model_paths(models_dir: Path, parameter: str) -> list[Path]:
    candidates: list[Path] = []
    for name in _candidate_model_names(parameter):
        candidates.append(models_dir / name)
    nested_dir = models_dir / "model_weights"
    if nested_dir.exists():
        for name in _candidate_model_names(parameter):
            candidates.append(nested_dir / name)
    return candidates


def _resolve_model_path(models_dir: Path, parameter: str) -> Path:
    for candidate in _candidate_model_paths(models_dir, parameter):
        if candidate.exists():
            return candidate

    expected = ", ".join(_candidate_model_names(parameter))
    raise FileNotFoundError(
        f"Missing CatRange model artifact for {parameter!r} in {models_dir}. "
        f"Expected one of: {expected}"
    )


class CatRangeInference:
    """Run CatRange kcat/KM bin prediction from raw sequence and SMILES."""

    def __init__(self, models_dir: str | Path, device: str = "auto", verbose: bool = True):
        _require_runtime()
        self.models_dir = Path(models_dir)
        _ensure_model_artifacts(self.models_dir)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.verbose = verbose
        self._esmc = None
        self._chem_tokenizer = None
        self._chem_model = None
        self._models: dict[str, object] = {}
        self._stats: dict[str, dict] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[CatRange] {message}", flush=True)

    def _resolve_model_path(self, models_dir: Path, parameter: str) -> Path:
        return _resolve_model_path(models_dir, parameter)

    def _load_esmc(self) -> None:
        if self._esmc is not None:
            return
        self._log("Loading ESM-C...")
        from esm.models.esmc import ESMC

        self._esmc = ESMC.from_pretrained("esmc_600m").to(self.device)
        self._esmc.eval()

    def _load_chemberta(self) -> None:
        if self._chem_model is not None:
            return
        self._log("Loading ChemBERTa...")
        from transformers import AutoModel, AutoTokenizer

        repo = "seyonec/PubChem10M_SMILES_BPE_450k"
        self._chem_tokenizer = AutoTokenizer.from_pretrained(repo)
        self._chem_model = AutoModel.from_pretrained(repo).to(self.device)
        self._chem_model.eval()

    def _load_model(self, parameter: str) -> None:
        parameter = parameter.lower()
        if parameter in self._models:
            return
        model_path = _resolve_model_path(self.models_dir, parameter)
        stats_path = self.models_dir / f"{parameter}_esmc_FINAL_stats.pt"
        self._models[parameter] = joblib.load(model_path)
        if stats_path.exists():
            self._stats[parameter] = torch.load(stats_path, map_location="cpu", weights_only=False)
        else:
            self._stats[parameter] = {}

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> np.ndarray:
        self._load_esmc()
        from esm.sdk.api import ESMProtein, LogitsConfig

        sequence = str(sequence).strip().upper()
        protein = ESMProtein(sequence=sequence)
        tokens = self._esmc.encode(protein)
        out = self._esmc.logits(tokens, LogitsConfig(sequence=True, structure=True, return_embeddings=True))
        reps = out.embeddings[0, 1 : len(sequence) + 1].float()
        return reps.mean(dim=0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_smiles(self, smiles: str) -> np.ndarray:
        self._load_chemberta()
        inputs = self._chem_tokenizer([str(smiles).strip()], return_tensors="pt", padding=True, truncation=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        out = self._chem_model(**inputs)
        return out.last_hidden_state[0].float().mean(dim=0).cpu().numpy().astype(np.float32)

    def embed_pairs(self, pairs: Iterable[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        seq_embeddings = []
        smiles_embeddings = []
        for sequence, smiles in pairs:
            seq_embeddings.append(self.embed_sequence(sequence))
            smiles_embeddings.append(self.embed_smiles(smiles))
        return np.stack(seq_embeddings), np.stack(smiles_embeddings)

    def _standardize(self, parameter: str, seq_embeddings: np.ndarray, smiles_embeddings: np.ndarray) -> np.ndarray:
        stats = self._stats.get(parameter, {})
        mean_1 = float(stats.get("mean_1", 0.0))
        std_1 = max(float(stats.get("std_1", 1.0)), 1e-8)
        mean_2 = float(stats.get("mean_2", 0.0))
        std_2 = max(float(stats.get("std_2", 1.0)), 1e-8)
        seq = (seq_embeddings - mean_1) / std_1
        sub = (smiles_embeddings - mean_2) / std_2
        return np.concatenate([seq, sub], axis=1).astype(np.float32)

    def predict_from_embeddings(
        self,
        seq_embeddings: np.ndarray,
        smiles_embeddings: np.ndarray,
        parameter: str = "kcat",
    ) -> pd.DataFrame:
        parameter = parameter.lower()
        if parameter not in BIN_EDGES:
            raise ValueError("parameter must be 'kcat' or 'km'")
        self._load_model(parameter)
        x = self._standardize(parameter, seq_embeddings, smiles_embeddings)
        model = self._models[parameter]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)
            pred_bin = probs.argmax(axis=1)
            confidence = probs.max(axis=1)
        else:
            pred_bin = model.predict(x).astype(int)
            probs = np.full((len(pred_bin), len(BIN_EDGES[parameter]) - 1), np.nan)
            confidence = np.full(len(pred_bin), np.nan)
        centers = _log_bin_centers(parameter)
        expected_log10 = probs @ centers if np.isfinite(probs).all() else np.full(len(pred_bin), np.nan)
        labels = _bin_labels(parameter)
        out = pd.DataFrame(
            {
                f"{parameter}_pred_bin": pred_bin.astype(int),
                f"{parameter}_pred_range": [labels[int(i)] for i in pred_bin],
                f"{parameter}_confidence": confidence,
                f"{parameter}_expected_log10": expected_log10,
            }
        )
        for idx in range(probs.shape[1]):
            out[f"{parameter}_prob_{idx}"] = probs[:, idx]
        return out

    def predict(self, pairs: Iterable[tuple[str, str]], parameter: str = "kcat") -> pd.DataFrame:
        pairs = list(pairs)
        seq_embeddings, smiles_embeddings = self.embed_pairs(pairs)
        out = self.predict_from_embeddings(seq_embeddings, smiles_embeddings, parameter=parameter)
        out.insert(0, "smiles", [smiles for _, smiles in pairs])
        out.insert(0, "sequence", [sequence for sequence, _ in pairs])
        return out


def _ensure_model_artifacts(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for parameter in ("kcat", "km"):
        if any(path.exists() for path in _candidate_model_paths(models_dir, parameter)):
            continue

        archive_path = models_dir / "model_weights.zip"
        if archive_path.exists():
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(models_dir)
            except zipfile.BadZipFile:
                archive_path.unlink(missing_ok=True)

        if not any(path.exists() for path in _candidate_model_paths(models_dir, parameter)):
            if not archive_path.exists():
                url = "https://huggingface.co/ssbio/CatRange/resolve/main/model_weights.zip"
                with urllib.request.urlopen(url, timeout=60) as response, open(archive_path, "wb") as fh:
                    shutil.copyfileobj(response, fh)
                try:
                    with zipfile.ZipFile(archive_path, "r") as zf:
                        zf.extractall(models_dir)
                except zipfile.BadZipFile as exc:
                    raise FileNotFoundError(
                        f"CatRange model archive is not a valid zip file: {archive_path}"
                    ) from exc

            if not any(path.exists() for path in _candidate_model_paths(models_dir, parameter)):
                raise FileNotFoundError(
                    f"CatRange model artifact missing after extraction for {parameter!r}: {models_dir}"
                )
