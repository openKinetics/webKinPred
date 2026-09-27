from api.methods.base import MethodDescriptor, SubprocessEngineConfig


descriptor = MethodDescriptor(
    key="CatRange",
    model_version="2",
    display_name="CatRange",
    authors=(
        "Karuna Anna Sajeevan, Abraham Osinuga, B Arunraj, Sakib Ferdous, "
        "Nabia Shahreen, Mohammed Sakib Noor, Shashank Koneru, Laura Mariana Santos-Correa, "
        "Rahil Salehi, Niaz Bahar Chowdhury, Randy Aryee, Brisa Calderon-Lopez, "
        "Supantha Dey, Ankur Mali, Rajib Saha, and Ratul Chowdhury"
    ),
    publication_title="CatRange Enables Robust Prediction of Enzyme Variant Kinetic Regimes",
    citation_url="https://doi.org/10.1093/pnasnexus/pgag309",
    repo_url="https://github.com/ssbio/CatRange",
    supports=["kcat", "Km"],
    input_format="single",
    output_cols={
        "kcat": "Predicted kcat range: kcat (1/s)",
        "Km": "Predicted KM range: kM (M)",
    },
    max_seq_len=1024,
    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT"},
        "Km": {"kinetics_type": "KM"},
    },
    # ESM-C protein embeddings are computed via the shared `esmc` env (GPU
    # offload with local CPU fallback), NOT inside catrange_env. ChemBERTa
    # substrate embeddings stay in catrange_env. See models/CatRange/predict.py.
    embeddings_used=["esmc"],
    subprocess=SubprocessEngineConfig(
        python_path_key="CatRange",
        script_key="CatRange",
        data_path_env={
            "CATRANGE_REPO_ROOT": "CatRange",
            "CATRANGE_MODELS_DIR": "CatRange",
            "CATRANGE_MEDIA_DIR": "media",
        },
        # CATRANGE_ESMC_PYTHON (shared esmc env, for the local embedding
        # fallback) is injected by the engine from PYTHON_PATHS at run time.
    ),
)
