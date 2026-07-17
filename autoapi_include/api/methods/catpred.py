# api/methods/catpred.py
#
# Method descriptor for CatPred.

from api.methods.base import MethodDescriptor, SubprocessEngineConfig


descriptor = MethodDescriptor(
    key="CatPred",
    display_name="CatPred",
    authors=("Veda Sheersh Boorla, Somtirtha Santra, Costas D. Maranas"),
    publication_title=(
        "CatPred: A comprehensive framework for deep learning in vitro enzyme kinetic parameters"
    ),
    citation_url="https://www.nature.com/articles/s41467-025-57215-9",
    repo_url="https://github.com/maranasgroup/CatPred",
    more_info=(
        "Recommended for wildtype proteins. CatPred kcat consumes the complete "
        "semicolon-separated substrate set natively; CatPred Km is predicted per substrate."
    ),
    supports=["kcat", "Km"],
    input_format="single",
    input_behavior_by_target={
        "kcat": "native_multi",
        "Km": "expanded_pair",
    },
    output_cols={
        "kcat": "kcat (1/s)",
        "Km": "KM (mM)",
    },
    max_seq_len=2048,
    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT"},
        "Km": {"kinetics_type": "KM"},
    },
    subprocess=SubprocessEngineConfig(
        python_path_key="CatPred",
        script_key="CatPred",
        fail_on_gpu_precompute_failure=False,
        data_path_env={
            "CATPRED_REPO_ROOT": "CatPred",
            "CATPRED_CHECKPOINT_ROOT": "CatPred_production_checkpoints",
            "CATPRED_MEDIA_PATH": "media",
            "CATPRED_TOOLS_PATH": "tools",
        },
        extra_env={
            "PROTEIN_EMBED_USE_CPU": "1",
        },
    ),
    embeddings_used=["esm2"],
)
