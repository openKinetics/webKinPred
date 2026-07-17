"""
Runtime path resolution for prediction engines.

Outside Docker, engines should read paths from ``config_local``.
Inside Docker, engines should read paths from ``config_docker``.
"""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType


def _is_docker_runtime() -> bool:
    """
    Detect whether the current process is running inside a Docker container.
    """
    return os.path.exists("/.dockerenv")


def _import_config_module() -> ModuleType:
    """
    Import the preferred config module for the current runtime.

    Preference:
    - Docker runtime  -> webKinPred.config_docker
    - Local runtime   -> webKinPred.config_local

    Falls back to the other module if the preferred one is missing.
    """
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


_cfg = _import_config_module()

PYTHON_PATHS: dict = _as_dict(getattr(_cfg, "PYTHON_PATHS", {}))
PREDICTION_SCRIPTS: dict = _as_dict(getattr(_cfg, "PREDICTION_SCRIPTS", {}))
DATA_PATHS: dict = _as_dict(getattr(_cfg, "DATA_PATHS", {}))
