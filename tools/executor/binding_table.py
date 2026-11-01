"""Read textual sequence identities and serialized residue scores without inference."""

from __future__ import annotations


def read_binding_table(path):
    import pandas as pd

    # A hexadecimal sequence ID can resemble an exponent (e.g. 0e6681122986).
    # Numeric inference both changes identities and crashes some pandas runtimes
    # on such exponents. Scores remain serialized until the weighting code reads
    # them; neither column is a scalar numeric feature.
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not {"PDB", "Pred_BS_Scores"} <= set(frame.columns):
        raise ValueError("Binding-site table requires sequence IDs and residue scores")
    if frame["PDB"].eq("").any() or frame["PDB"].duplicated().any():
        raise ValueError("Binding-site table requires unique nonempty sequence IDs")
    return frame
