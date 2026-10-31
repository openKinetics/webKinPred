"""Retention for verified executor artifacts, never legacy filesystem trees."""

from pathlib import PurePosixPath


def retention_for(identity):
    if identity.get("representation_version") == "execution-v2":
        parts = PurePosixPath(identity.get("path", "")).parts
        if "residue_vecs" in parts or (
            len(parts) >= 2
            and parts[:2]
            in {
                ("sequence_info", "esm1v"),
                ("sequence_info", "omniesi_esm2"),
                ("sequence_info", "iecata_prot_t5_residues"),
            }
        ):
            return "transient"
        # These are verified vector representations, including checkpoint-specific
        # CatPred attentive pooling. Do not infer residue shape from "ESM" alone.
        if (
            "mean_vecs" in parts
            or "weighted_vecs" in parts
            or (
                len(parts) >= 2
                and parts[:2]
                in {
                    ("sequence_info", "esm1b_turnup"),
                    ("sequence_info", "catpred_esm2"),
                    ("sequence_info", "esm2_layer_last_mean"),
                    ("sequence_info", "catrange_esmc"),
                }
            )
        ):
            return "permanent"
        if parts[:2] == ("sequence_info", "molecular"):
            return "permanent"
        return "protected"
    if identity.get("job_input") or identity.get("result") == "final":
        return "permanent"
    if identity.get("job") and (identity.get("stage") or identity.get("result")):
        return "cache"
    return "protected"
