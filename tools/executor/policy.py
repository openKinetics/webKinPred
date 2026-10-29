"""Small dependency-free helpers importable by the older ML environments."""

import json
import os
from pathlib import Path


def cache_only():
    return (
        os.environ.get("WEBKINPRED_EXECUTION_POLICY") == "gpu-only"
        and os.environ.get("WEBKINPRED_EMBED_WORKER") != "1"
    )


def require_cuda(torch):
    from tools.executor.instrumentation import install

    install(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_UNAVAILABLE: embedding worker requires CUDA")


def require_cached(condition, label):
    if cache_only() and not condition:
        raise RuntimeError("MISSING_EMBEDDING: " + label)


def planned_sequence_ids(sequences):
    path = os.environ.get("SEQMAP_MANIFEST_PATH")
    if not path:
        return None
    mapping = json.loads(Path(path).read_text())
    return [mapping[sequence] for sequence in sequences]


def propagate_resource_error(error):
    if os.environ.get("WEBKINPRED_EXECUTION_POLICY") != "gpu-only":
        return
    message = str(error).lower()
    if isinstance(error, (MemoryError, OSError, RuntimeError, ImportError)) or any(
        x in message
        for x in (
            "out of memory",
            "missing_embedding",
            "gpu_unavailable",
            "no space left",
            "resource temporarily unavailable",
        )
    ):
        raise error


_cursors = {}


class DeterministicEmbeddingInputError(ValueError):
    pass


def limit_estimator(model):
    """Bound inference parallelism without changing fitted parameters or weights."""
    if os.environ.get("WEBKINPRED_EXECUTION_POLICY") != "gpu-only":
        return model
    threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    if model.__class__.__module__.startswith("xgboost"):
        # XGBModel.set_params replays every saved training parameter into the
        # fitted booster. Older pickles can contain list-valued eval_metric,
        # which that path stringifies into an invalid metric name. Change only
        # the inference thread limit; preserve the fitted model configuration.
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        booster.set_param({"nthread": threads})
        if hasattr(model, "n_jobs"):
            model.n_jobs = threads
        return model
    if hasattr(model, "get_params") and hasattr(model, "set_params"):
        updates = {
            key: threads
            for key in model.get_params(deep=True)
            if key.rsplit("__", 1)[-1] in ("n_jobs", "nthread", "num_threads", "thread_count")
        }
        if updates:
            model.set_params(**updates)
    return model


def molecular_vectors(method, inputs):
    import numpy as np

    manifest_path = os.environ.get("WEBKINPRED_MOLECULAR_MANIFEST")
    if not manifest_path:
        raise RuntimeError("MISSING_EMBEDDING: molecular manifest")
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest[method]
    start = _cursors.get(method, 0)
    selected = rows[start : start + len(inputs)]
    if [row["input"] for row in selected] != list(inputs):
        raise RuntimeError("MISSING_EMBEDDING: molecular batch context mismatch")
    _cursors[method] = start + len(inputs)
    values = [
        np.load(
            Path(os.environ.get("WEBKINPRED_MEDIA_ROOT", "/app/media")) / row["path"],
            allow_pickle=False,
        )
        for row in selected
    ]
    for value in values:
        if value.dtype.kind == "U":
            raise DeterministicEmbeddingInputError(str(value[0]))
    return np.stack(values)
