from __future__ import annotations

from pathlib import Path
from typing import Any

from webKinPred.similarity_dataset_registry import SIMILARITY_DATASET_REGISTRY


SERVER_LIMIT = 10000
DEFAULT_ALLOWED_FRONTEND_IPS = ["127.0.0.1", "localhost"]


_DATA_PATH_REL = {
    "CatPred": "models/CatPred",
    "CatPred_production_checkpoints": "models/CatPred/data/pretrained/production",
    "DLKcat": "models/DLKcat/DeeplearningApproach/Data",
    "DLKcat_Results": "models/DLKcat/DeeplearningApproach/Results",
    "EITLEM": "models/EITLEM",
    "TurNup": "models/TurNup/data",
    "UniKP": "models/UniKP-main",
    "CataPro": "models/CataPro",
    "KinForm": "models/KinForm/results",
    "MMISA-KM": "models/MMISA-KM",
    "OmniESI": "models/OmniESI",
    "OmniESI-additional": "models/OmniESI/code/OmniESI_additional_data/additional_data",
    "OmniESI-embed": "media/sequence_info/omniesi_esm2",
    "OmniESI-weights": "cache/omniesi",
    "IECata": "models/IECata",
    "CatRange": "models/CatRange",
    "realkcat_esm2_last_mean": "media/sequence_info/esm2_layer_last_mean",
    "iecata_prot_t5_residues": "media/sequence_info/iecata_prot_t5_residues",
    "media": "media",
    "tools": "tools",
}


_PREDICTION_SCRIPT_REL = {
    "CatPred": "models/CatPred/catpred/integration/webkinpred_adapter.py",
    "DLKcat": "models/DLKcat/DeeplearningApproach/Code/example/prediction_for_input.py",
    "EITLEM": "models/EITLEM/Code/eitlem_prediction_script_batch.py",
    "TurNup": "models/TurNup/code/kcat_prediction_batch.py",
    "UniKP": "models/UniKP-main/run_unikp_batch.py",
    "CataPro": "models/CataPro/inference/custom_predict.py",
    "KinForm": "models/KinForm/code/main.py",
    "MMISA-KM": "models/MMISA-KM/upstream/script/prediction_script.py",
    "OmniESI": "models/OmniESI/batch_predict.py",
    "IECata": "models/IECata/prediction_script.py",
    "CatRange": "models/CatRange/predict.py",
}


def _join(base_path: str | Path, rel_path: str) -> str:
    return str((Path(base_path) / rel_path).resolve())


def build_data_paths(base_path: str | Path) -> dict[str, str]:
    return {key: _join(base_path, rel) for key, rel in _DATA_PATH_REL.items()}


def build_prediction_scripts(base_path: str | Path) -> dict[str, str]:
    return {key: _join(base_path, rel) for key, rel in _PREDICTION_SCRIPT_REL.items()}


def build_similarity_datasets(fastas_dir: str | Path) -> dict[str, dict[str, str | list[str]]]:
    fastas_dir = str(Path(fastas_dir).resolve())
    datasets: dict[str, dict[str, str | list[str]]] = {}
    for label, meta in SIMILARITY_DATASET_REGISTRY.items():
        meta_obj: dict[str, Any] = dict(meta)
        fasta_filename = str(meta_obj.get("fasta_filename", ""))
        db_name = str(meta_obj.get("db_name", ""))
        method_keys_obj = meta_obj.get("method_keys", [])
        if isinstance(method_keys_obj, list):
            method_keys = [str(key) for key in method_keys_obj]
        else:
            method_keys = []
        datasets[label] = {
            "label": label,
            "fasta": f"{fastas_dir}/{fasta_filename}",
            "target_db": f"{fastas_dir}/dbs/{db_name}",
            "method_keys": method_keys,
        }
    return datasets


def build_experimental_paths(media_dir: str | Path) -> tuple[str, str]:
    media_dir = Path(media_dir).resolve()
    km_csv = str(media_dir / "experimental" / "km_experimental.csv")
    kcat_csv = str(media_dir / "experimental" / "kcat_experimental.csv")
    return km_csv, kcat_csv
