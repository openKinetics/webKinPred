"""
api/utils/extra_info.py
───────────────────────
Utility for turning an experimental-lookup dict into the
'Extra Information' string shown in output.csv
"""

from typing import Any

# ------------------------------------------------------------------
# internal helpers
# ------------------------------------------------------------------
def _nullish(v: Any) -> bool:
    """True if *v* counts as 'not provided'."""
    return v is None or str(v).strip().lower() in {"", "nan", "none", "-"}

def _source(exp: dict) -> str:
    """Return BRENDA / Sabio-RK / UniProt or 'Experimental' as fallback."""
    if exp.get("from_brenda") == 1:
        return "BRENDA"
    if exp.get("from_sabio") == 1:
        return "Sabio-RK"
    if exp.get("from_uniprot") == 1:
        return "UniProt"
    return "Experimental"

# ------------------------------------------------------------------
# public function
# ------------------------------------------------------------------
def build_extra_info(exp: dict, param_type: str, prediction: str = '', model_key: str = '') -> str:
    """
    Parameters
    ----------
    exp : dict
        Record returned by `lookup_experimental`, already guaranteed to
        contain at least the keys used below.
    param_type : str
        'Km' or 'kcat' (case-insensitive).  Used to pick the SD field.
    prediction : str, optional
        The prediction value, if available. Used to include in the output.
    model_key : str, optional
        The model key used for the prediction, if applicable.

    Returns
    -------
    str
        A single human-readable paragraph, or '' if exp['found'] is False.
    """
    if not exp.get("found"):
        return ""

    src     = _source(exp)
    prot_id = exp.get("protein_ID", "unknown ID")
    parts   = [f"This is reported in {src} for protein {prot_id}"]

    # protein type / mutation
    descr_bits = []
    if not _nullish(exp.get("protein_type")):
        descr_bits.append(str(exp["protein_type"]))
    if not _nullish(exp.get("mutation")):
        mutation = exp["mutation"]
        if 'mutant' in mutation.lower():
            # remove 'mutant' from the description
            mutation = mutation.replace('mutant', '').strip()
        descr_bits.append(str(mutation))
    if descr_bits:
        parts[-1] += f" ({' '.join(descr_bits)})"

    parts[-1] += f" with {exp.get('substrate_inchi')}"

    # standard deviation
    sd_key = f"{param_type.lower()}_SD"
    sd_val = exp.get(sd_key)
    if _nullish(sd_val):
        parts.append("Standard deviation is not reported")
    else:
        parts.append(f"Standard deviation is {sd_val}")

    # full reaction, if both sides available
    if not _nullish(exp.get("all_substrates")) and not _nullish(exp.get("all_products")):
        parts.append(
            "Full reaction is:\n"
            f"{exp['all_substrates']} --> {exp['all_products']}"
        )

    # temperature / pH
    temp = None if _nullish(exp.get("temperature")) else exp["temperature"]
    ph   = None if _nullish(exp.get("ph"))         else exp["ph"]
    if temp is not None or ph is not None:
        t_str  = f"temperature {temp}" if temp is not None else ""
        ph_str = f"pH {ph}"            if ph   is not None else ""
        conj   = " and " if t_str and ph_str else ""
        parts.append(f"at {t_str}{conj}{ph_str}".strip())
    
    # prediction value
    if prediction:
        parts.append(f"Prediction by {model_key} is {prediction}")

    return ". ".join(parts)
