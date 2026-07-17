from api.methods.base import MethodDescriptor, SubprocessEngineConfig

descriptor = MethodDescriptor(
    key="IECata",
    display_name="IECata",
    authors=("Jingjing Wang, Yanpeng Zhao, Zhijiang Yang, Ge Yao, Penggang Han, " \
    "Jiajia Liu, Chang Chen, Peng Zan, Xiukun Wan, Xiaochen Bo, Hui Jiang"),
    publication_title=(
        "IECata: Interpretable bilinear attention network and evidential "
        "deep learning improve the catalytic efficiency prediction of enzymes"
    ),
    citation_url="https://doi.org/10.1093/bib/bbaf283",
    repo_url="https://github.com/zhaoyanpeng208/IECata",

    supports=["kcat/Km"],
    input_format="single",
    output_cols={"kcat/Km": "kcat/Km (1/(s·M))"},
    max_seq_len=1000,   # IECata pads to 1200; stay under to avoid truncation

    col_to_kwarg={"Substrate": "substrates"},
    target_kwargs={"kcat/Km": {}},

    subprocess=SubprocessEngineConfig(
        python_path_key="IECata",
        script_key="IECata",
        data_path_env={
            "IECATA_DATA":      "IECata",
            # IECATA_EMBED_DIR is the directory where the GPU worker writes
            # the ephemeral per-residue .npy files before this subprocess runs.
            "IECATA_EMBED_DIR": "iecata_prot_t5_residues",
        },
    ),

    # Declares the embedding family so the generic engine triggers the
    # GPU precompute step before calling the subprocess.
    embeddings_used=["iecata_prot_t5_residues"],
)
