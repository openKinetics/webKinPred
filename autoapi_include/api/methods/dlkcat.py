# api/methods/dlkcat.py
#
# Method descriptor for DLKcat.

from api.methods.base import MethodDescriptor


def _dlkcat_predictions_lazy(*args, **kwargs):
    # Import prediction engine only when a job is actually executed.
    from api.prediction_engines.dlkcat import dlkcat_predictions

    return dlkcat_predictions(*args, **kwargs)


descriptor = MethodDescriptor(
    key="DLKcat",
    display_name="DLKcat",
    authors=(
        "Feiran Li, Le Yuan, Hongzhong Lu, Gang Li, Yu Chen, "
        "Martin K. M. Engqvist, Eduard J. Kerkhoven & Jens Nielsen"
    ),
    publication_title=(
        "Deep learning-based kcat prediction enables improved "
        "enzyme-constrained model reconstruction"
    ),
    citation_url="https://www.nature.com/articles/s41929-022-00798-z",
    repo_url="https://github.com/SysBioChalmers/DLKcat",
    more_info="",
    # ── Capabilities ──────────────────────────────────────────────────────────
    supports=["kcat"],
    input_format="single",
    output_cols={"kcat": "kcat (1/s)"},
    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len=float("inf"),
    # ── Input mapping ─────────────────────────────────────────────────────────
    # The framework reads df["Substrate"] for each valid row and passes it to
    # dlkcat_predictions() as the `substrates` keyword argument.
    col_to_kwarg={"Substrate": "substrates"},
    # ── Per-target extra kwargs ───────────────────────────────────────────────
    # DLKcat predicts kcat only; no extra flags are needed.
    target_kwargs={
        "kcat": {},
    },
    pred_func=_dlkcat_predictions_lazy,
    # ── Embeddings ────────────────────────────────────────────────────────────
    # DLKcat uses its own graph neural network for substrate representation
    # and a domain-adapted LSTM for protein sequences.  Neither is exposed
    # as a shared embedding in our infrastructure.
    embeddings_used=[],
)
