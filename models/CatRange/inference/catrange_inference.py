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


class CatRangeInputError(ValueError):
    """An unsupported input row that does not invalidate neighboring rows."""


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
    ]


def _candidate_model_paths(models_dir: Path, parameter: str) -> list[Path]:
    # OKP's existing CatRange setting points at models/CatRange. Also accept
    # an explicit artifact directory, as used by standalone inference.
    directories = (
        models_dir,
        models_dir / "model_weights",
        models_dir / "inference" / "models",
        models_dir / "inference" / "models" / "model_weights",
    )
    # Prefer the released ESM-C weights across all supported layouts. The
    # archive's unversioned *_model.pkl files use ESM-2 (2048 features), which
    # is incompatible with this pipeline's ESM-C + ChemBERTa features (1920).
    return [directory / name for name in _candidate_model_names(parameter)
            for directory in directories]


def _resolve_model_path(models_dir: Path, parameter: str) -> Path:
    for candidate in _candidate_model_paths(models_dir, parameter):
        if candidate.exists():
            return candidate

    expected = ", ".join(_candidate_model_names(parameter))
    raise FileNotFoundError(
        f"Missing CatRange model artifact for {parameter!r} in {models_dir}. "
        f"Expected one of: {expected}"
    )


def _resolve_stats_path(models_dir: Path, model_path: Path, parameter: str) -> Path:
    # Release archives may place weights in model_weights/, while the tracked
    # training statistics live directly in inference/models/.
    name = f"{parameter}_esmc_FINAL_stats.pt"
    candidates = [
        model_path.parent / name,
        models_dir / name,
        models_dir / "inference" / "models" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing required CatRange standardization statistics for {parameter!r}. "
        f"Searched: {', '.join(str(path) for path in candidates)}. "
        "Use the statistics released with the selected model weights."
    )


def _validate_stats(stats: object, stats_path: Path) -> dict[str, float]:
    if not isinstance(stats, dict):
        raise ValueError(f"Invalid CatRange standardization statistics in {stats_path}: expected a dict")
    validated: dict[str, float] = {}
    for key in ("mean_1", "std_1", "mean_2", "std_2"):
        if key not in stats:
            raise ValueError(f"Invalid CatRange standardization statistics in {stats_path}: missing {key}")
        try:
            value = np.asarray(stats[key])
            if value.ndim != 0 or value.dtype.kind not in "iuf":
                raise ValueError("expected a numeric scalar")
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid CatRange standardization statistic {key} in {stats_path}: expected a numeric scalar"
            ) from exc
        if not np.isfinite(number) or (key.startswith("std_") and number <= 0):
            raise ValueError(
                f"Invalid CatRange standardization statistic {key} in {stats_path}: "
                "means must be finite and standard deviations must be finite and positive"
            )
        validated[key] = number
    return validated


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
        stats_path = _resolve_stats_path(self.models_dir, model_path, parameter)
        stats = _validate_stats(
            torch.load(stats_path, map_location="cpu", weights_only=False), stats_path
        )
        model = joblib.load(model_path)
        self._stats[parameter] = stats
        self._models[parameter] = model

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
        text = str(smiles).strip()
        if not text:
            raise CatRangeInputError("CatRange requires a non-empty substrate SMILES")
        try:
            inputs = self._chem_tokenizer(
                [text], return_tensors="pt", padding=True, truncation=False
            )
        except (ValueError, IndexError) as exc:
            raise CatRangeInputError(f"CatRange could not tokenize the substrate: {exc}") from exc
        token_count = int(inputs["input_ids"].shape[-1])
        token_limit = self._smiles_token_limit()
        if token_count > token_limit:
            raise CatRangeInputError(
                f"Substrate is too long for CatRange's ChemBERTa encoder "
                f"({token_count} tokens including special tokens; maximum {token_limit})"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        out = self._chem_model(**inputs)
        vector = out.last_hidden_state[0].float().mean(dim=0).cpu().numpy().astype(np.float32)
        if not np.isfinite(vector).all():
            raise CatRangeInputError("CatRange produced a non-finite substrate embedding")
        return vector

    def _smiles_token_limit(self) -> int:
        config = self._chem_model.config
        limit = int(config.max_position_embeddings)
        if config.model_type == "roberta":
            # RoBERTa assigns non-padding positions starting at padding_idx+1.
            # A 512-entry position table with padding_idx=1 supports 510 tokens,
            # including the leading/trailing special tokens.
            padding_idx = int(self._chem_model.embeddings.padding_idx)
            limit -= padding_idx + 1
        tokenizer_limit = self._chem_tokenizer.model_max_length
        if tokenizer_limit is not None:
            limit = min(limit, int(tokenizer_limit))
        if limit <= 0:
            raise RuntimeError("CatRange ChemBERTa has an invalid token-limit configuration")
        return limit

    def embed_pairs(self, pairs: Iterable[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        seq_embeddings = []
        smiles_embeddings = []
        for sequence, smiles in pairs:
            seq_embeddings.append(self.embed_sequence(sequence))
            smiles_embeddings.append(self.embed_smiles(smiles))
        return np.stack(seq_embeddings), np.stack(smiles_embeddings)

    def _standardize(self, parameter: str, seq_embeddings: np.ndarray, smiles_embeddings: np.ndarray) -> np.ndarray:
        stats = self._stats[parameter]
        mean_1 = stats["mean_1"]
        std_1 = stats["std_1"]
        mean_2 = stats["mean_2"]
        std_2 = stats["std_2"]
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
