import os

from webKinPred.config_base import (
    DEFAULT_ALLOWED_FRONTEND_IPS,
    SERVER_LIMIT,
    build_data_paths,
    build_experimental_paths,
    build_prediction_scripts,
    build_similarity_datasets,
)


BASE_PATH = "/app"
DATA_PATH = BASE_PATH
FASTAS_DIR = f"{BASE_PATH}/fastas"

PYTHON_PATHS = {
    "CatPred": "/opt/conda/envs/catpred_env/bin/python",
    "DLKcat": "/opt/conda/envs/dlkcat_env/bin/python",
    "EITLEM": "/opt/conda/envs/eitlem_env/bin/python",
    "TurNup": "/opt/conda/envs/turnup_env/bin/python",
    "UniKP": "/opt/conda/envs/unikp/bin/python",
    "CataPro": "/opt/conda/envs/catapro_env/bin/python",
    "KinForm": "/opt/conda/envs/kinform_env/bin/python",
    "MMISA-KM": "/opt/conda/envs/mmisakm_env/bin/python",
    "OmniESI": "/opt/conda/envs/omniesi_env/bin/python",
    "esm2": "/opt/conda/envs/esm/bin/python",
    "esmc": "/opt/conda/envs/esmc/bin/python",
    "t5": "/opt/conda/envs/prot_t5/bin/python",
    "pseq2sites": "/opt/conda/envs/pseq2sites/bin/python",
    "IECata": "/opt/conda/envs/iecata_env/bin/python",
    "CatRange": "/opt/conda/envs/catrange_env/bin/python",
}

DATA_PATHS = build_data_paths(BASE_PATH)
PREDICTION_SCRIPTS = build_prediction_scripts(BASE_PATH)

CATPRED_ROOT = os.environ.get("WEBKINPRED_CATPRED_ROOT")
if CATPRED_ROOT:
    DATA_PATHS["CatPred"] = CATPRED_ROOT
    DATA_PATHS["CatPred_production_checkpoints"] = f"{CATPRED_ROOT}/data/pretrained/production"
    PREDICTION_SCRIPTS["CatPred"] = f"{CATPRED_ROOT}/catpred/integration/webkinpred_adapter.py"

OMNIESI_ADDITIONAL_DATA = os.environ.get("WEBKINPRED_OMNIESI_ADDITIONAL_DATA")
if OMNIESI_ADDITIONAL_DATA:
    DATA_PATHS["OmniESI-additional"] = OMNIESI_ADDITIONAL_DATA
    
SIMILARITY_DATASETS = build_similarity_datasets(FASTAS_DIR)
TARGET_DBS = {label: item["target_db"] for label, item in SIMILARITY_DATASETS.items()}

CONDA_PATH = os.environ.get("WEBKINPRED_CONDA_PATH")
if not CONDA_PATH:
    default_conda_path = "/opt/conda/bin/conda"
    CONDA_PATH = default_conda_path if os.path.exists(default_conda_path) else None
DEBUG = True
ALLOWED_FRONTEND_IPS = [*DEFAULT_ALLOWED_FRONTEND_IPS, "frontend", "backend"]

KM_CSV, KCAT_CSV = build_experimental_paths(f"{BASE_PATH}/media")
