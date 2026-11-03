"""Describe historical resident-model families across bounded work-unit sizes.

This metadata helper is not an admission gate. Admission uses each individually
validated profile and the resource ledger, independent of mixture measurements.

Only sequence count may grow. The new work unit must itself be profiled; its full
RAM, VRAM and disk reservation is still charged by admission. Molecular encoders
and any change to image, model settings or per-sequence shape cannot inherit.
"""

PROTEIN_STEPS = frozenset(
    {
        "prot_t5_mean",
        "eitlem_esm1v",
        "omniesi_esm2",
        "catrange_esmc",
        "turnup_esm1b",
        "catpred_embed",
        "iecata_prot_t5_residues",
        "kinform_t5_residues",
        "kinform_esm2_residues",
        "kinform_esmc_residues",
        "kinform_pseq2sites",
    }
)


def families(profiles):
    profiles = {p.fingerprint: p for p in profiles if p.validated and not p.suspended}
    result = {}
    for child in sorted(profiles.values(), key=lambda p: p.limits.get("sequences", 0)):
        if child.stage_kind != "embedding" or not child.resources.get("vram"):
            continue
        if child.settings.get("step") not in PROTEIN_STEPS:
            continue
        settings = {
            k: v
            for k, v in child.settings.items()
            if k not in {"cohort_sources", "cohort_parent_image"}
        }
        for ident in child.settings.get("cohort_sources", []):
            parent = profiles.get(ident)
            if (
                parent is None
                or parent.stage_kind != child.stage_kind
                or parent.method != child.method
            ):
                continue
            if parent.image != child.image or not parent.resources.get("vram"):
                continue
            parent_settings = {
                k: v
                for k, v in parent.settings.items()
                if k not in {"cohort_sources", "cohort_parent_image"}
            }
            if settings != parent_settings or set(child.limits) != set(parent.limits):
                continue
            if child.limits.get("sequences", 0) <= parent.limits.get("sequences", 0):
                continue
            if any(child.limits[k] > parent.limits[k] for k in child.limits if k != "sequences"):
                continue
            result[child.fingerprint] = result.get(parent.fingerprint, parent.fingerprint)
            break
    return result
