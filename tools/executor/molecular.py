"""CUDA molecular encoders, including compatibility for legacy batch positions."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from tools.executor.policy import require_cuda


def module_at(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "WordVocab"):
        # Published legacy vocabularies were pickled from an executable module.
        # Resolve the same class when loading them through an isolated adapter.
        sys.modules["__main__"].WordVocab = module.WordVocab
    return module


def legacy_transformer_vectors(model, vocab, split, rows, torch, device="cuda"):
    """Reproduce the existing batch-axis position values without resetting on retry."""
    tokens = []
    for row in rows:
        parts = split(row["input"]).split()
        if len(parts) > 218:
            parts = parts[:109] + parts[-109:]
        values = [3] + [vocab.stoi.get(p, 1) for p in parts] + [2]
        tokens.append(values + [0] * (220 - len(values)))
    src = torch.tensor(tokens, device=device).t()
    positions = torch.tensor([row["legacy_position"] for row in rows], device=device)
    with torch.no_grad():
        output = model.embed(src) + model.pe.pe[:, positions]
        for layer in list(model.trfm.encoder.layers)[:-1]:
            output = layer(output, None)
        penultimate = output.detach().cpu().numpy()
        output = model.trfm.encoder.layers[-1](output, None)
        if model.trfm.encoder.norm:
            output = model.trfm.encoder.norm(output)
        final = output.detach().cpu().numpy()
    return np.hstack([final.mean(axis=0), final.max(axis=0), final[0], penultimate[0]])


def main():
    import torch

    require_cuda(torch)
    os.environ.setdefault("UNIKP_MEDIA_PATH", os.environ["WEBKINPRED_MEDIA_ROOT"])
    os.environ.setdefault("UNIKP_TOOLS_PATH", "/app/tools")
    request = json.loads(Path(sys.argv[1]).read_text())
    payload = request["payload"]
    rows = payload["rows"]
    method = payload["molecular_method"]
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if method == "UniKP":
        module = module_at(Path("/app/models/UniKP-main/run_unikp_batch.py"), "execution_unikp")
        vocab, model = module.load_smiles_model()
        model = model.to("cuda").eval()
        size = max(1, int(request["settings"]["batch_size"]))
        vectors = np.concatenate(
            [
                legacy_transformer_vectors(model, vocab, module.split, rows[i : i + size], torch)
                for i in range(0, len(rows), size)
            ]
        )
    elif method == "KinForm":
        root = Path("/app/models/KinForm/code")
        sys.path.insert(0, str(root))
        from smiles_embeddings.smiles_transformer.build_vocab import WordVocab
        from smiles_embeddings.smiles_transformer.pretrain_trfm import TrfmSeq2seq
        from smiles_embeddings.smiles_transformer.split_util import split

        sys.modules["__main__"].WordVocab = WordVocab
        vocab = WordVocab.load_vocab(str(root / "smiles_embeddings/smiles_transformer/vocab.pkl"))
        model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
        model.load_state_dict(
            torch.load(
                root / "smiles_embeddings/smiles_transformer/trfm_12_23000.pkl", map_location="cpu"
            )
        )
        model = model.to("cuda").eval()
        size = max(1, int(request["settings"]["batch_size"]))
        vectors = np.concatenate(
            [
                legacy_transformer_vectors(model, vocab, split, rows[i : i + size], torch)
                for i in range(0, len(rows), size)
            ]
        )
    elif method == "CataPro":
        module = module_at(
            Path("/app/models/CataPro/inference/custom_predict.py"), "execution_catapro"
        )
        os.environ["WEBKINPRED_MOLECULAR_FP32"] = "1"
        vectors = module.get_molt5_embeddings(
            [row["input"] for row in rows],
            Path("/app/models/CataPro/models/molt5-base-smiles2caption"),
            torch.device("cuda"),
            batch_size=request["settings"].get("batch_size", 1),
        )
    elif method == "CatRange":
        module = module_at(
            Path("/app/models/CatRange/inference/catrange_inference.py"), "execution_catrange"
        )
        model = module.CatRangeInference(
            "/app/models/CatRange/inference/models",
            device="cuda",
            verbose=False,
            encoder_only=True,
        )
        media = Path(os.environ["WEBKINPRED_MEDIA_ROOT"])
        for row in rows:
            path = media / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                vector = model.embed_smiles(row["input"])
            except module.CatRangeInputError as exc:
                # Explicit scientific input rejection, not an infrastructure/cache failure.
                np.save(path, np.asarray([str(exc)]))
            else:
                np.save(path, vector.astype(np.float32))
        return
    else:
        raise ValueError("Unknown molecular embedding representation")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite molecular embedding")
    media = Path(os.environ["WEBKINPRED_MEDIA_ROOT"])
    for row, vector in zip(rows, vectors):  # noqa: B905 - legacy ML Python < 3.10
        path = media / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector.astype(np.float32))


if __name__ == "__main__":
    main()
