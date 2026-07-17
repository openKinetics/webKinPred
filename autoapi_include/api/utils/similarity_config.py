"""
Runtime-aware config access for MMseqs similarity analysis.
"""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType


def _is_docker_runtime() -> bool:
    return os.path.exists("/.dockerenv")


def _import_runtime_config() -> ModuleType:
    preferred = "webKinPred.config_docker" if _is_docker_runtime() else "webKinPred.config_local"
    fallback = (
        "webKinPred.config_local"
        if preferred.endswith("config_docker")
        else "webKinPred.config_docker"
    )
    try:
        return import_module(preferred)
    except Exception:
        return import_module(fallback)


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


_cfg = _import_runtime_config()

CONDA_PATH = getattr(_cfg, "CONDA_PATH", None)
SIMILARITY_DATASETS: dict = _as_dict(getattr(_cfg, "SIMILARITY_DATASETS", {}))
TARGET_DBS: dict = _as_dict(getattr(_cfg, "TARGET_DBS", {}))

# Fallback compatibility for older configs that only expose TARGET_DBS.
if not SIMILARITY_DATASETS and TARGET_DBS:
    SIMILARITY_DATASETS = {
        label: {"label": label, "target_db": path, "fasta": None}
        for label, path in TARGET_DBS.items()
    }
