# api/methods/unikp.py
#
# Method descriptor for UniKP.

from api.methods.base import MethodDescriptor


def _unikp_predictions_lazy(*args, **kwargs):
    # Import prediction engine only when a job is actually executed.
    from api.prediction_engines.unikp import unikp_predictions

    return unikp_predictions(*args, **kwargs)


descriptor = MethodDescriptor(
    key="UniKP",
    display_name="UniKP",
    authors=("Han Yu, Huaxiang Deng, Jiahui He, Jay D. Keasling & Xiaozhou Luo"),
    publication_title=(
        "UniKP: a unified framework for the prediction of enzyme kinetic parameters"
    ),
    citation_url="https://www.nature.com/articles/s41467-023-44113-1",
    repo_url="https://github.com/Luo-SynBioLab/UniKP",
    more_info="",
    # ── Capabilities ──────────────────────────────────────────────────────────
    supports=["kcat", "Km"],
    input_format="single",
    output_cols={
        "kcat": "kcat (1/s)",
        "Km": "KM (mM)",
    },
    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len=1000,
    # ── Input mapping ─────────────────────────────────────────────────────────
    col_to_kwarg={"Substrate": "substrates"},
    # ── Per-target extra kwargs ───────────────────────────────────────────────
    # unikp_predictions() accepts a `kinetics_type` argument to switch between
    # kcat and KM prediction modes.
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT"},
        "Km": {"kinetics_type": "KM"},
    },
    pred_func=_unikp_predictions_lazy,
    # ── Embeddings ────────────────────────────────────────────────────────────
    # UniKP uses ProtT5-XL-UniRef50 for protein representation internally.
    # The ProtT5 model is also available as a shared embedding in our
    # infrastructure (key: "prot_t5"), though UniKP bundles its own copy
    # within the unikp conda environment.
    embeddings_used=["prot_t5"],
)
