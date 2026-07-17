# api/methods/mmisa_km.py
#
# Method descriptor for MMISA-KM.

from api.methods.base import MethodDescriptor, SubprocessEngineConfig


descriptor = MethodDescriptor(
    key="MMISA-KM",
    display_name="MMISA-KM",
    authors="Aijie Song, Kai Wang",
    publication_title=(
        "MMISA-KM: A multimodal deep learning model for Michaelis constant prediction"
    ),
    citation_url="https://doi.org/10.1109/DDCLS66240.2025.11064981",
    repo_url="https://github.com/kaiwang-group/MMISA-KM",
    more_info=(
        ""
    ),
    supports=["Km"],
    input_format="single",
    output_cols={
        "Km": "KM (mM)",
    },
    max_seq_len=500,
    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={
        "Km": {"kinetics_type": "KM"},
    },
    subprocess=SubprocessEngineConfig(
        python_path_key="MMISA-KM",
        script_key="MMISA-KM",
    ),
    embeddings_used=[],
)
