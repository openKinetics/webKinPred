# api/methods/eitlem.py
#
# Method descriptor for EITLEM-Kinetics.

from api.methods.base import MethodDescriptor


def _eitlem_predictions_lazy(*args, **kwargs):
    # Import prediction engine only when a job is actually executed.
    from api.prediction_engines.eitlem import eitlem_predictions

    return eitlem_predictions(*args, **kwargs)


descriptor = MethodDescriptor(
    key="EITLEM",
    display_name="EITLEM-Kinetics",
    authors=("Xiaowei Shen, Ziheng Cui, Jianyu Long, Shiding Zhang, Biqiang Chen, Tianwei Tan"),
    publication_title=(
        "EITLEM-Kinetics: A deep-learning framework for kinetic "
        "parameter prediction of mutant enzymes"
    ),
    citation_url=("https://www.sciencedirect.com/science/article/pii/S2667109324002665"),
    repo_url="https://github.com/XvesS/EITLEM-Kinetics",
    more_info="",
    # ── Capabilities ──────────────────────────────────────────────────────────
    supports=["kcat", "Km"],
    input_format="single",
    output_cols={
        "kcat": "kcat (1/s)",
        "Km": "KM (mM)",
    },
    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len=1024,
    # ── Input mapping ─────────────────────────────────────────────────────────
    col_to_kwarg={"Substrate": "substrates"},
    # ── Per-target extra kwargs ───────────────────────────────────────────────
    # eitlem_predictions() accepts a `kinetics_type` argument to switch between
    # kcat and KM prediction modes.
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT"},
        "Km": {"kinetics_type": "KM"},
    },
    pred_func=_eitlem_predictions_lazy,
    # ── Embeddings ────────────────────────────────────────────────────────────
    # EITLEM uses ESM-1b for protein embeddings and RDKit Morgan fingerprints
    # for substrates.  These are bundled in the eitlem_env conda environment
    # and are not exposed as shared embeddings.
    embeddings_used=[],
)
