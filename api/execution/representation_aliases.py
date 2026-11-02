"""Reuse identical encoders without rewriting historical artifacts or requests.

These pairs route the listed representations through the very same worker code.
Prediction heads never share identities. Every inventory, precision, context and
preprocessing field must match; only the two legacy method labels may differ.
"""

from copy import deepcopy
from pathlib import PurePosixPath

KINFORM_ROOTS = frozenset(
    {
        "prot_t5_layer_19",
        "prot_t5_last",
        "esm2_layer_26",
        "esm2_layer_29",
        "esmc_layer_24",
        "esmc_layer_32",
    }
)


def alternate_identity(identity):
    if (
        identity.get("representation_version") != "execution-v2"
        or identity.get("target") is not None
    ):
        return None
    method = identity.get("method")
    models = identity.get("models", {})
    if not isinstance(models, dict) or models.get("method") != method:
        return None
    if models.get("representation_policy") != "20261028-v2":
        return None
    parts = PurePosixPath(identity.get("path", "")).parts
    if method in {"OmniESI", "OmniESI-O2DENet"}:
        if len(parts) < 3 or parts[:2] != ("sequence_info", "omniesi_esm2"):
            return None
        other = "OmniESI-O2DENet" if method == "OmniESI" else "OmniESI"
    elif method in {"KinForm-H", "KinForm-L"}:
        protein = (
            len(parts) >= 4
            and parts[0] == "sequence_info"
            and parts[1] in KINFORM_ROOTS
            and parts[2] in {"residue_vecs", "mean_vecs", "weighted_vecs"}
        )
        molecular = len(parts) >= 4 and parts[:3] == ("sequence_info", "molecular", "KinForm")
        binding = len(parts) >= 2 and parts[0] == "pseq2sites"
        if not (protein or molecular or binding):
            return None
        other = "KinForm-L" if method == "KinForm-H" else "KinForm-H"
    else:
        return None
    alternate = deepcopy(identity)
    alternate["method"] = other
    alternate["models"]["method"] = other
    return alternate
