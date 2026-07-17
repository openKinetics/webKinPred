# api/methods/kinform_h.py
#
# Method descriptor for KinForm-H (high sequence-similarity variant).

from api.methods.base import MethodDescriptor


def _kinform_predictions_lazy(*args, **kwargs):
    # Import prediction engine only when a job is actually executed.
    from api.prediction_engines.kinform import kinform_predictions

    return kinform_predictions(*args, **kwargs)


descriptor = MethodDescriptor(
    key="KinForm-H",
    display_name="KinForm-H",
    authors="Saleh Alwer, Ronan M T Fleming",
    publication_title=(
        "KinForm: Kinetics Informed Feature Optimised Representation "
        "Models for Enzyme kcat and KM Prediction"
    ),
    citation_url="https://www.nature.com/articles/s41540-026-00692-5",
    repo_url="https://github.com/Digital-Metabolic-Twin-Centre/KinForm",
    more_info="Recommended for proteins with high sequence similarity to training data.",
    # ── Capabilities ──────────────────────────────────────────────────────────
    supports=["kcat", "Km"],
    input_format="single",
    output_cols={
        "kcat": "kcat (1/s)",
        "Km": "KM (mM)",
    },
    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len=1500,
    # ── Input mapping ─────────────────────────────────────────────────────────
    col_to_kwarg={"Substrate": "substrates"},
    # ── Per-target extra kwargs ───────────────────────────────────────────────
    # kinform_predictions() accepts both `kinetics_type` and `model_variant`.
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT", "model_variant": "H"},
        "Km": {"kinetics_type": "KM", "model_variant": "H"},
    },
    pred_func=_kinform_predictions_lazy,
    # ── Embeddings ────────────────────────────────────────────────────────────
    # KinForm uses four protein embedding models that are all available in our
    # shared embedding infrastructure (see api/embeddings/registry.py).
    # A new method that needs any of these can reuse the existing conda envs
    # by reading the python paths from PYTHON_PATHS in config.
    embeddings_used=["esm2", "esmc", "prot_t5", "pseq2sites"],
)
