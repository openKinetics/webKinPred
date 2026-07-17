# api/methods/kinform_l.py
#
# Method descriptor for KinForm-L (low sequence-similarity variant).

from api.methods.base import MethodDescriptor


def _kinform_predictions_lazy(*args, **kwargs):
    # Import prediction engine only when a job is actually executed.
    from api.prediction_engines.kinform import kinform_predictions

    return kinform_predictions(*args, **kwargs)


descriptor = MethodDescriptor(
    key="KinForm-L",
    display_name="KinForm-L",
    authors="Saleh Alwer, Ronan M T Fleming",
    publication_title=(
        "KinForm: Kinetics Informed Feature Optimised Representation "
        "Models for Enzyme kcat and KM Prediction"
    ),
    citation_url="https://www.nature.com/articles/s41540-026-00692-5",
    repo_url="https://github.com/Digital-Metabolic-Twin-Centre/KinForm",
    more_info="Recommended for proteins with low sequence similarity to training data.",
    # ── Capabilities ──────────────────────────────────────────────────────────
    # KinForm-L supports kcat prediction only; it is not available for KM.
    supports=["kcat"],
    input_format="single",
    output_cols={"kcat": "kcat (1/s)"},
    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len=1500,
    # ── Input mapping ─────────────────────────────────────────────────────────
    col_to_kwarg={"Substrate": "substrates"},
    # ── Per-target extra kwargs ───────────────────────────────────────────────
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT", "model_variant": "L"},
    },
    pred_func=_kinform_predictions_lazy,
    # ── Embeddings ────────────────────────────────────────────────────────────
    embeddings_used=["esm2", "esmc", "prot_t5", "pseq2sites"],
)
