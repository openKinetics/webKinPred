from api.methods.base import MethodDescriptor, SubprocessEngineConfig


descriptor = MethodDescriptor(
    key="CatRange",
    display_name="CatRange",
    authors=(
        "Karuna Anna Sajeevan, Abraham Osinuga, Arunraj B, Sakib Ferdous, "
        "Nabia Shahreen, Shashank Koneru, Laura Mariana Santos-Correa, "
        "Rahil Salehi, Niaz Bahar Chowdhury, Randy Aryee, Brisa Calderon-Lopez, "
        "Supantha Dey, Ankur Mali, Rajib Saha, and Ratul Chowdhury"
    ),
    publication_title="CatRange Enables Robust Prediction of Enzyme Variant Kinetic Regimes",
    citation_url="https://doi.org/10.1101/2025.02.10.637555",
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
    subprocess=SubprocessEngineConfig(
        python_path_key="CatRange",
        script_key="CatRange",
        data_path_env={
            "CATRANGE_REPO_ROOT": "CatRange",
            "CATRANGE_MODELS_DIR": "CatRange",
        },
    ),
)
