# models/MMISA-KM/upstream/script/prediction_script.py
"""
MMISA-KM batch prediction script for webKinPred generic subprocess engine.

Contract:
    python prediction_script.py --input <input.json> --output <output.json>

Input JSON:
    {"rows": [{"sequence": "MKTAY...", "substrates": "CC(=O)O"}, ...]}

Output JSON:
    {"predictions": [12.5, null, 0.045, ...], "invalid_indices": [1, ...]}

Environment variables (all optional — fallbacks are safe for local dev):
    MMISA_KM_ROOT               Root override; used only to locate trained_model.pt
                                (local dev shortcut).
    MMISA_KM_CONTACT_MAP_SOURCE Contact-map source: sequential|esmfold|alphafold|template
                                Default: sequential
    CUDA_VISIBLE_DEVICES        Standard mechanism to select a specific GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

# ---------------------------------------------------------------------------
# Resolve upstream script directory and inject into sys.path.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ---------------------------------------------------------------------------
# RDKit InChI helper — API location changed between rdkit versions.
# ---------------------------------------------------------------------------
try:
    from rdkit.Chem.inchi import MolFromInchi as _mol_from_inchi
except ImportError:
    _mol_from_inchi = getattr(Chem, "MolFromInchi", None)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# MMISA-KM source module imports.
# ---------------------------------------------------------------------------
from model import GNNNet  # noqa: E402
from data_process import (  # noqa: E402
    atom_features,
    residue_features,
    encode_smiles,
    encode_sequence,
    ELEMENT_LIST,  # noqa: F401
)
from contactmap import generate_contact_map  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Mappings
# ============================================================================

_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

_AA_TO_ID: dict[str, int] = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "K": 9,  "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, "X": 21,
}

_CHAR_SMI_SET: dict[str, int] = {
    "(": 1,   ".": 2,   "0": 3,   "2": 4,   "4": 5,   "6": 6,   "8": 7,
    "@": 8,   "B": 9,   "D": 10,  "F": 11,  "H": 12,  "L": 13,  "N": 14,
    "P": 15,  "R": 16,  "T": 17,  "V": 18,  "Z": 19,  "\\": 20, "b": 21,
    "d": 22,  "f": 23,  "h": 24,  "l": 25,  "n": 26,  "r": 27,  "t": 28,
    "#": 29,  "%": 30,  ")": 31,  "+": 32,  "-": 33,  "/": 34,  "1": 35,
    "3": 36,  "5": 37,  "7": 38,  "9": 39,  "=": 40,  "A": 41,  "C": 42,
    "E": 43,  "G": 44,  "I": 45,  "K": 46,  "M": 47,  "O": 48,  "S": 49,
    "U": 50,  "W": 51,  "Y": 52,  "[": 53,  "]": 54,  "a": 55,  "c": 56,
    "e": 57,  "g": 58,  "i": 59,  "m": 60,  "o": 61,  "s": 62,  "u": 63,
    "y": 64,
}

MODEL_CONFIG: dict[str, Any] = {
    "embed_dim":        256,
    "n_output":         1,
    "num_features_pro": 33,
    "num_features_mol": 78,
    "dropout":          0.2,
}

MAX_SMILES_LEN = 100
MAX_SEQ_LEN    = 500

def _resolve_weights_path() -> Path:
    """Locate trained_model.pt via env var or relative-to-script fallback."""
    mmisa_root = os.getenv("MMISA_KM_ROOT")
    if mmisa_root:
        candidate = Path(mmisa_root).resolve() / "script" / "trained_model.pt"
        if candidate.exists():
            return candidate
    return _THIS_DIR / "trained_model.pt"


def _contact_map_source() -> str:
    return os.getenv("MMISA_KM_CONTACT_MAP_SOURCE", "sequential")


# ============================================================================
# Argument Parsing
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMISA-KM webKinPred batch adapter")
    parser.add_argument("--input",  required=True, help="Path to JSON input payload")
    parser.add_argument("--output", required=True, help="Path to JSON output payload")
    return parser.parse_args()


# ============================================================================
# Input Validation & Normalisation
# ============================================================================

def _safe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Input payload is malformed: 'rows' must be a list")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Input payload is malformed: row {i} is not an object")
    return list(rows)


def _normalise_substrate(raw: Any) -> str | None:
    """
    Normalise substrate input to a canonical SMILES string.

    Accepts a single SMILES string, semicolon-separated string with one token,
    a list with one token, or an InChI string (converted via RDKit).
    Returns None for multi-substrate inputs or unparseable strings.
    """
    if isinstance(raw, list):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        text = str(raw).strip()
        if not text:
            return None
        tokens = [tok.strip() for tok in text.split(";") if tok.strip()]

    if len(tokens) != 1:
        return None

    substrate = tokens[0]

    mol = Chem.MolFromSmiles(substrate)
    if mol is not None:
        return Chem.MolToSmiles(mol)

    if _mol_from_inchi is not None:
        mol = _mol_from_inchi(substrate)
        if mol is not None:
            return Chem.MolToSmiles(mol)

    return None


# ============================================================================
# Graph Construction
# ============================================================================

def _build_mol_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x_rows: list[np.ndarray] = []
    for atom in mol.GetAtoms():
        feat = atom_features(atom)
        feat_sum = feat.sum()
        if feat_sum > 1e-8:
            feat = feat / feat_sum
        x_rows.append(feat)

    x = torch.tensor(np.stack(x_rows), dtype=torch.float32)

    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((a, b))
        edges.append((b, a))
    for i in range(mol.GetNumAtoms()):
        edges.append((i, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def _build_pro_graph(
    sequence: str,
    contact_map_source: str = "sequential",
) -> Data:
    seq = sequence[:MAX_SEQ_LEN].upper()

    features: list[np.ndarray] = []
    for aa in seq:
        onehot = np.zeros(21, dtype=np.float32)
        aa_idx = max(0, min(20, _AA_TO_ID.get(aa, 21) - 1))
        onehot[aa_idx] = 1.0
        props = residue_features(aa)
        features.append(np.concatenate([onehot, props]))  # 21 + 12 = 33

    if not features:
        features = [np.zeros(33, dtype=np.float32)]

    x = torch.tensor(np.stack(features), dtype=torch.float32)

    contacts = generate_contact_map(
        sequence=seq,
        structure_source=contact_map_source,
    )

    n = min(len(features), contacts.shape[0])
    edges: list[tuple[int, int]] = [
        (i, j) for i in range(n) for j in range(n) if contacts[i, j] > 0.5
    ]

    existing_self_loops: set[int] = {e[0] for e in edges if e[0] == e[1]}
    for i in range(n):
        if i not in existing_self_loops:
            edges.append((i, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


# ============================================================================
# Model Loading
# ============================================================================

def _load_model(device: torch.device) -> GNNNet:
    weights_path = _resolve_weights_path()
    if not weights_path.exists():
        raise RuntimeError(
            f"MMISA-KM model weights not found at: {weights_path}. "
            "Set MMISA_KM_ROOT to the upstream directory containing "
            "'script/trained_model.pt'."
        )
    model = GNNNet(**MODEL_CONFIG)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded MMISA-KM weights from %s on %s", weights_path, device)
    return model


# ============================================================================
# Per-Row Inference with Progress Streaming
# ============================================================================

def _predict_rows(rows: list[dict[str, Any]]) -> tuple[list[Any], list[int]]:
    """
    Run MMISA-KM inference row-by-row with live progress streaming.
    """
    n_total = len(rows)
    predictions: list[Any] = [None] * n_total
    invalid_indices: list[int] = []
    progress_emitted = -1

    def _emit_progress(done: int) -> None:
        nonlocal progress_emitted
        if done > progress_emitted:
            progress_emitted = done
            print(f"Progress: {done}/{n_total}", flush=True)

    cmap_source = _contact_map_source()
    logger.info("MMISA-KM contact-map source: %s", cmap_source)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("MMISA-KM inference device: %s", device)

    try:
        model = _load_model(device)
    except Exception as exc:
        if os.environ.get("WEBKINPRED_EXECUTION_POLICY") == "gpu-only":
            raise
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        logger.error("Model load failed — marking all rows invalid: %s", exc)
        invalid_indices = list(range(n_total))
        print(f"Progress: {n_total}/{n_total}", flush=True)
        return predictions, invalid_indices

    for idx, row in enumerate(rows):
        try:
            sequence = str(row.get("sequence", "")).strip().upper()
            if not sequence:
                raise ValueError("Empty sequence")
            invalid_aas = [ch for ch in sequence if ch not in _VALID_AAS]
            if invalid_aas:
                raise ValueError(
                    f"Sequence contains non-standard residues: {set(invalid_aas)}"
                )

            substrate = _normalise_substrate(
                row.get("substrates", row.get("substrate", ""))
            )
            if substrate is None:
                raise ValueError("Substrate could not be parsed or is multi-substrate")

            mol_graph = _build_mol_graph(substrate)
            if mol_graph is None:
                raise ValueError(f"RDKit could not build graph for SMILES: {substrate!r}")

            smi_enc = encode_smiles(substrate, MAX_SMILES_LEN, _CHAR_SMI_SET)
            seq_enc = encode_sequence(sequence, MAX_SEQ_LEN, _AA_TO_ID)

            pro_graph = _build_pro_graph(
                sequence,
                contact_map_source=cmap_source,
            )

            smi_t = torch.tensor(smi_enc[np.newaxis], dtype=torch.long, device=device)
            seq_t = torch.tensor(seq_enc[np.newaxis], dtype=torch.long, device=device)
            mol_b = Batch.from_data_list([mol_graph]).to(device)
            pro_b = Batch.from_data_list([pro_graph]).to(device)

            with torch.no_grad():
                raw_output = model(smi_t, seq_t, mol_b, pro_b)

            pred_log = float(raw_output.view(-1).detach().cpu()[0])

            if math.isnan(pred_log) or math.isinf(pred_log):
                raise ValueError(f"Non-finite model output ({pred_log}) at row {idx}")

            predictions[idx] = float(10.0 ** pred_log)

        except Exception as exc:
            from tools.executor.policy import propagate_resource_error
            propagate_resource_error(exc)
            logger.warning("Row %d failed: %s", idx, exc)
            invalid_indices.append(idx)
            predictions[idx] = None

        _emit_progress(idx + 1)

    _emit_progress(n_total)

    return predictions, sorted(set(invalid_indices))


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    args = _parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = _safe_rows(payload)
    logger.info("MMISA-KM: processing %d input rows", len(rows))

    predictions, invalid_indices = _predict_rows(rows)

    result = {
        "predictions":    predictions,
        "invalid_indices": invalid_indices,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh)

    n_valid = len(rows) - len(invalid_indices)
    logger.info(
        "MMISA-KM: complete — %d/%d valid predictions written to %s",
        n_valid, len(rows), output_path,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        from tools.executor.policy import propagate_resource_error
        propagate_resource_error(exc)
        print(f"[MMISA-KM] FATAL: {exc}", file=sys.stderr, flush=True)
        raise