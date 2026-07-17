# api/methods/turnup.py
#
# Method descriptor for TurNup.

from api.methods.base import MethodDescriptor


def _turnup_predictions_lazy(*args, **kwargs):
    # Import prediction engine only when a job is actually executed.
    from api.prediction_engines.turnup import turnup_predictions

    return turnup_predictions(*args, **kwargs)


descriptor = MethodDescriptor(
    key="TurNup",
    display_name="TurNup",
    authors=("Alexander Kroll, Yvan Rousset, Xiao-Pan Hu, Nina A. Liebrand & Martin J. Lercher"),
    publication_title=(
        "Turnover number predictions for kinetically uncharacterised "
        "enzymes using machine and deep learning"
    ),
    citation_url="https://www.nature.com/articles/s41467-023-39840-4",
    repo_url="https://github.com/AlexanderKroll/Kcat_prediction",
    more_info="Recommended for natural reactions of wild-type enzymes.",
    # ── Capabilities ──────────────────────────────────────────────────────────
    supports=["kcat"],
    # TurNup uses the full-reaction CSV format: "Substrates" and "Products"
    # columns containing semicolon-separated SMILES or InChI strings.
    input_format="multi",
    output_cols={"kcat": "kcat (1/s)"},
    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len=1024,
    # ── Input mapping ─────────────────────────────────────────────────────────
    # The framework reads df["Substrates"] and df["Products"] for each valid
    # row and passes them to turnup_predictions() as `substrates` and `products`.
    col_to_kwarg={
        "Substrates": "substrates",
        "Products": "products",
    },
    # ── Per-target extra kwargs ───────────────────────────────────────────────
    # TurNup predicts kcat only; no extra flags are needed.
    target_kwargs={
        "kcat": {},
    },
    pred_func=_turnup_predictions_lazy,
    # ── Embeddings ────────────────────────────────────────────────────────────
    # TurNup uses ESM-1v for protein embeddings and reaction fingerprints
    # for substrate/product representation, both bundled in turnup_env.
    embeddings_used=[],
)
