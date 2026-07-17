# api/methods/registry.py
#
# Auto-discovering registry for MethodDescriptor objects.
#
# All modules inside api/methods/ that expose a module-level `descriptor`
# variable of type MethodDescriptor are loaded automatically.  No explicit
# registration is needed — just drop a new file in this directory.

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.methods.base import MethodDescriptor


# ---------------------------------------------------------------------------
# Internal registry store
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, "MethodDescriptor"] = {}
_loaded = False


def _load() -> None:
    """
    Walk all modules inside the api.methods package and collect descriptors.

    Skips the ``base`` and ``registry`` modules themselves.  Any module that
    defines a ``descriptor`` attribute at the top level is registered under
    ``descriptor.key``.
    """
    global _loaded
    if _loaded:
        return

    import api.methods as pkg

    for _finder, module_name, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if module_name in ("base", "registry"):
            continue
        mod = importlib.import_module(f"api.methods.{module_name}")
        if hasattr(mod, "descriptor"):
            desc = mod.descriptor
            if desc.key in _REGISTRY:
                raise ValueError(
                    f"Duplicate method key '{desc.key}' found in "
                    f"api/methods/{module_name}.py. "
                    f"Each method must have a unique key."
                )
            _REGISTRY[desc.key] = desc

    _loaded = True


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def get(key: str) -> "MethodDescriptor":
    """
    Return the descriptor for the given method key.

    Raises KeyError if no method with that key is registered.
    """
    _load()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown prediction method '{key}'. Available methods: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def all_methods() -> dict[str, "MethodDescriptor"]:
    """Return a copy of the full registry dict, keyed by method key."""
    _load()
    return dict(_REGISTRY)


def methods_supporting(target: str) -> list["MethodDescriptor"]:
    """
    Return all registered methods that support the given prediction target.

    Parameters
    ----------
    target : str
        ``"kcat"``, ``"Km"``, or ``"kcat/Km"``.
    """
    _load()
    return [desc for desc in _REGISTRY.values() if target in desc.supports]


def get_model_limits() -> dict[str, int | float]:
    """
    Return a dict mapping each registered method key to its max_seq_len.

    Used by validation utilities that need per-model sequence length limits.
    """
    _load()
    return {key: desc.max_seq_len for key, desc in _REGISTRY.items()}
