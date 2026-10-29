"""Small source-only release context; never packages model weights or user media."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOTS = ("api", "tools", "webKinPred", "db_models")

# Deployment adapters must be included after they have been committed, too.
MODEL_ADAPTERS = (
    "models/CatPred/catpred/integration/catpred_embed_gpu.py",
    "models/CatPred/catpred/integration/webkinpred_adapter.py",
    "models/CatRange/inference/catrange_inference.py",
    "models/CatRange/predict.py",
    "models/CataPro/inference/custom_predict.py",
    "models/DLKcat/DeeplearningApproach/Code/example/prediction_for_input.py",
    "models/EITLEM/Code/eitlem_prediction_script_batch.py",
    "models/IECata/prediction_script.py",
    "models/KinForm/code/main.py",
    "models/KinForm/code/protein_embeddings/prot_embeddings.py",
    "models/KinForm/code/protein_embeddings/t5_embeddings.py",
    "models/KinForm/code/pseq2sites/get_sites.py",
    "models/KinForm/code/pseq2sites/pseq2sites_stream_worker.py",
    "models/KinForm/code/utils/compute_embs.py",
    "models/KinForm/code/utils/smiles_features.py",
    "models/MMISA-KM/upstream/script/prediction_script.py",
    "models/OmniESI/batch_predict.py",
    "models/TurNup/code/enzyme_representations.py",
    "models/TurNup/code/kcat_prediction_batch.py",
    "models/UniKP-main/run_unikp_batch.py",
)


def create(root, destination):
    files = []
    for name in ROOTS:
        files.extend(p for p in (root / name).rglob("*.py") if "__pycache__" not in p.parts)
    files.extend(
        root / name
        for name in (
            "manage.py",
            "deploy.sh",
            "requirements.app.txt",
            "requirements.executor.txt",
            "Dockerfile.executor",
            "Dockerfile.execution-runtime",
            "Dockerfile.gpu-native",
            "docker-compose.prod.yml",
            "docker-compose.execution.yml",
            "docker-compose.executor-host.yml",
            "docs/resource-aware-execution.md",
            "docs/execution-profiling-2026-10-28.md",
            "docs/execution-deployment-2026-10-29.md",
        )
    )
    files.extend(p for p in (root / "deploy/execution").rglob("*") if p.is_file())
    files.extend(
        p
        for p in (root / "frontend").rglob("*")
        if p.is_file()
        and not {"node_modules", "build", "dist", ".git"}.intersection(p.parts)
        and not p.name.startswith(".env")
    )
    modified = subprocess.check_output(
        ["git", "diff", "--name-only", "--", "models"], cwd=root, text=True
    ).splitlines()
    patches = [
        root / name
        for name in sorted(set(modified) | set(MODEL_ADAPTERS))
        if name.endswith(".py") and (root / name).is_file()
    ]
    manifest = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files + patches
        },
        "model_patches": [str(p.relative_to(root)) for p in patches],
    }
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(files + patches):
            relative = str(path.relative_to(root))
            name = "patches/" + relative if path in patches else relative
            info = archive.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        data = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("release.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    create(Path.cwd(), args.destination)
