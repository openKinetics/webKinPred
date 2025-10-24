import os
from pathlib import Path


# Python bin paths
ESM_BIN = "/home/saleh/anaconda3/envs/esm/bin/python"
ESMC_BIN = "/home/saleh/anaconda3/envs/esmc/bin/python"
T5_BIN = "/home/saleh/anaconda3/envs/prot_t5/bin/python"
PSEQ2SITES_BIN = "/home/saleh/anaconda3/envs/pseq2sites/bin/python"
# ───────────────────────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent.parent
CONDA_EXE = Path(os.environ.get("CONDA_EXE", ""))
if not CONDA_EXE or str(CONDA_EXE) == "":
    # common user path fallback (non-fatal)
    candidate = ROOT / "anaconda3" / "bin" / "conda"
    if candidate.exists():
        CONDA_EXE = candidate
    else:
        CONDA_EXE = Path("/usr/bin/conda")
MMSEQS_ENV  = "mmseqs2_env"
DATA_DIR     = ROOT
RAW_DLKCAT     = DATA_DIR / "data/dlkcat_raw.json"
BS_PRED_PATH = DATA_DIR / "results/binding_sites/binding_sites_all.tsv"
CAT_PRED_DF = DATA_DIR / "results/catalytic_sites/cat_sites.csv"
SEQ_LOOKUP   = DATA_DIR / "results/sequence_id_to_sequence.pkl"
COMPUTED_EMBEDDINGS_PATHS = {
    "esm2": [ROOT / "results/protein_embeddings/esm2_layer_26", ROOT / "results/protein_embeddings/esm2_layer_29"],
    "esmc": [ROOT / "results/protein_embeddings/esmc_layer_24", ROOT / "results/protein_embeddings/esmc_layer_32"],
    "t5"  : [ROOT / "results/protein_embeddings/prot_t5_layer_19", ROOT / "results/protein_embeddings/prot_t5_last"],
}
# ---- Prot representation configs ----
EMB_COMBOS = {
    "ESMC":               dict(use_t5=False, use_esmc=True,  use_esm2=False),
    "T5":                 dict(use_t5=True,  use_esmc=False, use_esm2=False),
    "ESM2":              dict(use_t5=False, use_esmc=False, use_esm2=True),
    "ESMC+T5":            dict(use_t5=True,  use_esmc=True,  use_esm2=False),
    "T5+ESM2":            dict(use_t5=True,  use_esmc=False, use_esm2=True),
    "ESMC+ESM2":         dict(use_t5=False, use_esmc=True,  use_esm2=True),
    "ESMC+T5+ESM2":      dict(use_t5=True,  use_esmc=True,  use_esm2=True),
}

REP_MODES = {
    "global":        dict(prot_rep_mode="global"),
    "binding":       dict(prot_rep_mode="binding"),
    "binding+global": dict(prot_rep_mode="binding+global"),
}

CONFIGS = []
for emb_name, emb_flags in EMB_COMBOS.items():
    for rep_name, rep_flags in REP_MODES.items():
        name = f"{emb_name}|{rep_name}"
        CONFIGS.append(dict(name=name, **emb_flags, **rep_flags))

# ---- PCA configs ----
PCA_VALUES = [100, 200, 300, 400, 500, 750, 1000, 1750, None]  # None = no PCA
# PCA_VALUES = [10, None]  

CONFIGS_PCA = []

for emb_name, emb_flags in {
    "ESMC+ESM2+T5": dict(use_esmc=True, use_esm2=True, use_t5=True, t5_last_layer=True),
    "ESMC+T5":    dict(use_esmc=True, use_esm2=False, use_t5=True,t5_last_layer=True),
    
}.items():
    for n_comps in PCA_VALUES:
        use_pca = n_comps is not None
        label = (
            f"{emb_name}|+PCA|k={n_comps}"
            if use_pca else
            f"{emb_name}|-PCA"
        )
        CONFIGS_PCA.append(dict(
            name=label,
            use_pca=use_pca,
            n_comps=n_comps,
            prot_rep_mode="binding+global",
            **emb_flags
        ))

# ---- SMILES representations ----
SMILES_REPS = ['smiles_transformer', 'MFP', 'UniMol', 'FARM', 'molformer',
        'TopologicalTorsion', 'MinHash', 'MACCS', 'AtomPair', 'Avalon']

CONFIGS_SMILES_KCAT = [
    dict(
        name=f"{method}|PCA (ESMC+ESM2+T5, k=300)",
        smiles_method=method,
        use_pca=True,
        n_comps=300,
        use_esmc=True, use_esm2=True, use_t5=True,
        prot_rep_mode="binding+global",
        t5_last_layer=True,
    )
    for method in SMILES_REPS
] 
CONFIGS_SMILES_KM = [
    dict(
        name=f"{method}|ESMC+ESM2+T5 (no PCA)",
        smiles_method=method,
        use_pca=False,
        n_comps=None,
        prot_rep_mode="binding+global",
        use_esmc=True,
        use_esm2=True,
        use_t5=True,
        t5_last_layer=True,
    )
    for method in SMILES_REPS
]
# ---- Configs for KinForm-L and KinForm-H ----
CONFIG_L = dict(
    name="KinForm-L",
    use_pca=True,
    n_comps=300,
    prot_rep_mode="binding+global",
    use_esmc=True,
    use_esm2=True,
    use_t5=True,
    t5_last_layer=True,
    et_params=dict(
        n_estimators=100,)
)
CONFIG_H = dict(
    name="KinForm-H",
    use_pca=False,
    n_comps=None,
    prot_rep_mode="binding+global",
    use_esmc=True,
    use_esm2=True,
    use_t5=True,
    t5_last_layer=True,
    et_params=dict(
        n_estimators=100)
)
CONFIG_UniKP = dict(
    name="UniKP",
    use_pca=False,
    n_comps=None,
    prot_rep_mode="global",
    use_esmc=False,
    use_esm2=False,
    use_t5=True,
    t5_last_layer=True,
)

# ---- Configs for Ablation Study ----
CONFIG_BASE = CONFIG_UniKP.copy()
CONFIG_BASE.update(dict(name="Base (UniKP)"))

CONFIG_PLUS_ESMC = CONFIG_BASE.copy()
CONFIG_PLUS_ESMC.update(dict(use_esmc=True, name="+ESMC"))

CONFIG_PLUS_ESM2 = CONFIG_PLUS_ESMC.copy()
CONFIG_PLUS_ESM2.update(dict(use_esm2=True, name="+ESMC+ESM2"))

CONFIG_BINDING = CONFIG_PLUS_ESM2.copy()
CONFIG_BINDING.update(dict(prot_rep_mode="binding", name="+ESMC+ESM2+BindingAverage"))

CONFIG_BONDING_PLUS_GLOBAL = CONFIG_BINDING.copy()
CONFIG_BONDING_PLUS_GLOBAL.update(dict(prot_rep_mode="binding+global", name="+ESMC+ESM2+BindingAverage+GlobalAverage (KinForm-H)"))

CONFIG_PLUS_PCA = CONFIG_BONDING_PLUS_GLOBAL.copy()
CONFIG_PLUS_PCA.update(dict(use_pca=True, n_comps=300, name="+ESMC+ESM2+BindingAverage+GlobalAverage+PCA (KinForm-L)"))

CONFIGS_ABLATION = [
    CONFIG_BASE,
    CONFIG_PLUS_ESMC,
    CONFIG_PLUS_ESM2,
    CONFIG_BINDING,
    CONFIG_BONDING_PLUS_GLOBAL,
    CONFIG_PLUS_PCA
]