import os
from pathlib import Path

from webKinPred.config_base import (
    DEFAULT_ALLOWED_FRONTEND_IPS,
    SERVER_LIMIT,
    build_data_paths,
    build_experimental_paths,
    build_prediction_scripts,
    build_similarity_datasets,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = str(REPO_ROOT)
DATA_PATH = BASE_PATH
FASTAS_DIR = str(REPO_ROOT / "fastas")

CONDA_ENVS_DIR = Path(
    os.environ.get("WEBKINPRED_CONDA_ENVS_DIR", str(Path.home() / "anaconda3" / "envs"))
)


def _env_python(env_name: str) -> str:
    return str(CONDA_ENVS_DIR / env_name / "bin" / "python")


PYTHON_PATHS = {
    "CatPred": _env_python("catpred_env"),
    "DLKcat": _env_python("dlkcat_env"),
    "EITLEM": _env_python("eitlem_env"),
    "TurNup": _env_python("turnup_env"),
    "UniKP": _env_python("unikp"),
    "CataPro": _env_python("catapro_env"),
    "KinForm": _env_python("kinform_env"),
    "MMISA-KM": _env_python("mmisakm_env"),
    "OmniESI": os.environ.get("WEBKINPRED_OMNIESI_PYTHON", _env_python("omniesi_env")),
    "IECata": _env_python("iecata_env"),
    "CatRange": _env_python("catrange_env"),
    "esm2": _env_python("esm"),
    "esmc": _env_python("esmc"),
    "t5": _env_python("prot_t5"),
    "pseq2sites": _env_python("pseq2sites"),
}

DATA_PATHS = build_data_paths(BASE_PATH)
PREDICTION_SCRIPTS = build_prediction_scripts(BASE_PATH)

CATPRED_ROOT = os.environ.get("WEBKINPRED_CATPRED_ROOT")
if CATPRED_ROOT:
    catpred_root = str(Path(CATPRED_ROOT).resolve())
    DATA_PATHS["CatPred"] = catpred_root
    DATA_PATHS["CatPred_production_checkpoints"] = str(
        Path(catpred_root) / "data" / "pretrained" / "production"
    )
    PREDICTION_SCRIPTS["CatPred"] = str(
        Path(catpred_root) / "catpred" / "integration" / "webkinpred_adapter.py"
    )
    PYTHON_PATHS["CatPred"] = os.environ.get(
        "WEBKINPRED_CATPRED_PYTHON",
        _env_python("catpred_env"),
    )

SIMILARITY_DATASETS = build_similarity_datasets(FASTAS_DIR)
TARGET_DBS = {label: item["target_db"] for label, item in SIMILARITY_DATASETS.items()}

CONDA_PATH = os.environ.get(
    "WEBKINPRED_CONDA_PATH", str(Path.home() / "anaconda3" / "bin" / "conda")
)

ALLOWED_FRONTEND_IPS = [*DEFAULT_ALLOWED_FRONTEND_IPS, "140.203.228.102"]
DEBUG = True

KM_CSV, KCAT_CSV = build_experimental_paths(REPO_ROOT / "media")
