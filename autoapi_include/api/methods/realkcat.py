from api.methods.base import MethodDescriptor, SubprocessEngineConfig

descriptor = MethodDescriptor(
    key="RealKcat",
    display_name="RealKcat",
    authors="Sajeevan AK, Osinuga A, Mali A, et al.",
    publication_title="Robust Prediction of Enzyme Variant Kinetics with RealKcat",
    citation_url="https://doi.org/10.1101/2025.02.10.637555",
    repo_url="https://github.com/TKAI-LAB-Mali/RealKcat",
    
    supports=["kcat", "Km"],
    input_format="single",
    output_cols={"kcat": "kcat (1/s)", "Km": "Km (M)"},
    max_seq_len=1022,  # ESM2 context limit
    
    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT"},
        "Km": {"kinetics_type": "KM"}
    },
    
    subprocess=SubprocessEngineConfig(
        python_path_key="RealKcat",
        script_key="RealKcat",
        data_path_env={
            "REALKCAT_DATA": "RealKcat_DATA",
            "REALKCAT_EMBED_CACHE_DIR": "realkcat_esm2_last_mean",
        },
    ),

    embeddings_used=["realkcat_esm2_last_mean"],
)
