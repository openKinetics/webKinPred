# api/methods/catapro.py
#
# Method descriptor for CataPro.

from api.methods.base import MethodDescriptor, SubprocessEngineConfig


descriptor = MethodDescriptor(
    key="CataPro",
    display_name="CataPro",
    authors=(
        "Zechen Wang, Dongqi Xie, Dong Wu, Xiaozhou Luo, Sheng Wang, "
        "Yangyang Li, Yanmei Yang, Weifeng Li, Liangzhen Zheng"
    ),
    publication_title=("Robust enzyme discovery and engineering with deep learning using CataPro"),
    citation_url="https://www.nature.com/articles/s41467-025-58038-4",
    repo_url="https://github.com/zchwang/CataPro",
    more_info="",
    supports=["kcat", "Km", "kcat/Km"],
    input_format="single",
    output_cols={
        "kcat": "kcat (1/s)",
        "Km": "KM (mM)",
        "kcat/Km": "kcat/Km (1/(s*mM))",
    },
    max_seq_len=1000,
    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={
        "kcat": {"kinetics_type": "KCAT"},
        "Km": {"kinetics_type": "KM"},
        "kcat/Km": {"kinetics_type": "KCAT/KM"},
    },
    subprocess=SubprocessEngineConfig(
        python_path_key="CataPro",
        script_key="CataPro",
        data_path_env={
            "CATAPRO_MEDIA_PATH": "media",
            "CATAPRO_TOOLS_PATH": "tools",
            "CATAPRO_MODEL_ROOT": "CataPro",
        },
    ),
    embeddings_used=["prot_t5"],
)
