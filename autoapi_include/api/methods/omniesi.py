# api/methods/omniesi.py
#
# Method descriptor for OmniESI.

from api.methods.base import MethodDescriptor, SubprocessEngineConfig


descriptor = MethodDescriptor(
    key="OmniESI",
    display_name="OmniESI",
    authors="Zhiwei Nie, Hongyu Zhang, Hao Jiang, Yutian Liu, Xiansong Huang, Fan Xu, Jie Fu, Zhixiang Ren, Yonghong Tian, Wen-Bin Zhang, Jie Chen",
    publication_title=(
        "OmniESI: A unified framework for enzyme-substrate interaction prediction with progressive conditional deep learning"
    ),
    citation_url="https://doi.org/10.48550/arXiv.2506.17963",
    repo_url="https://github.com/Hong-yu-Zhang/OmniESI",
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
       "kcat": {"kinetics_type": "KCAT"},
        "Km": {"kinetics_type": "KM"}
    },
    subprocess=SubprocessEngineConfig(
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

