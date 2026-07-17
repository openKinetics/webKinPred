# api/methods/omniesi_o2denet.py
#
# Method descriptor for OmniESI + O2DENet.
#
# O2DENet is a training-time-only module (pseudodata augmentation + invariant
# representation learning) that leaves the OmniESI inference architecture
# unchanged.  The published "OmniESI + O2DENet" kcat/Km models therefore run on
# the existing OmniESI runtime (env, script, ESM2 embedding cache); only the
# trained weights differ.  The shared batch_predict.py wrapper selects the
# O2DENet checkpoints via the `variant` param wired through target_kwargs below,
# loading a single checkpoint per target from the fold_O2DENet weight folders.

from api.methods.base import MethodDescriptor, SubprocessEngineConfig


descriptor = MethodDescriptor(
    key="OmniESI-O2DENet",
    display_name="OmniESI + O2DENet",
    authors="Haomin Wu, Zhiwei Nie, Hongyu Zhang, Zhixiang Ren",
    publication_title=(
        "Pseudodata-Guided Invariant Representation Learning Boosts the "
        "Out-of-Distribution Generalization in Enzymatic Kinetic Parameter Prediction"
    ),
    citation_url="https://pubs.acs.org/doi/10.1021/acs.jcim.5c03204?ref=PDF",
    repo_url="https://github.com/blackjack534/O2DENet",
    more_info=(
        ""
    ),
    supports=["kcat", "Km"],
    input_format="single",
    output_cols={
        "kcat": "kcat (1/s)",
        "Km": "KM (mM)"
    },
    max_seq_len=1000,
    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT", "variant": "o2denet"},
        "Km":   {"kinetics_type": "KM",   "variant": "o2denet"},
    },
    subprocess=SubprocessEngineConfig(
        # Reuse the OmniESI conda env, prediction script, and shared esm2 cache.
        python_path_key="OmniESI",
        script_key="OmniESI",
        data_path_env={
            "OMNIESI_EMBED_CACHE_DIR": "OmniESI-embed",
            "OmniESI_CACHE_DIR":       "OmniESI-weights",
            "OmniESI_ADDITIONAL_DATA": "OmniESI-additional",
        },
    ),
    embeddings_used=["esm2"],
)
