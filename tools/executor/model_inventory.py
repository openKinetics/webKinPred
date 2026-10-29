"""Fingerprint installed model code, weights, tokenizers and environment metadata.

Run against preserved release trees, never against writable job directories. The
manifest records resolved symlink targets and fails if files change while read.
"""

import argparse
import hashlib
import json
from pathlib import Path

from api.execution.artifacts import atomic_json, sha256_file

SUFFIXES = {
    "",  # Some published checkpoints (including DLKcat) have no suffix.
    ".py",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".bin",
    ".model",
    ".ckpt",
    ".joblib",
    ".safetensors",
    ".json",
    ".yaml",
    ".yml",
    ".npy",
    ".npz",
    ".txt",
    ".vocab",
    ".csv",
    ".tsv",
}
IGNORE = {".git", "__pycache__", ".pytest_cache", "wandb", "logs"}


def fingerprint(roots):
    files = {}
    for label, directory in sorted(roots.items()):
        root = Path(directory).resolve(strict=True)
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if IGNORE.intersection(relative.parts) or not path.is_file():
                continue
            if path.suffix.lower() not in SUFFIXES and path.name not in {"METADATA", "RECORD"}:
                continue
            before = path.stat()
            checksum = sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("Model file changed during inventory: " + str(path))
            files[label + "/" + str(relative)] = {
                "source": str(path),
                "resolved": str(path.resolve()),
                "size": after.st_size,
                "sha256": checksum,
            }
    if not files:
        raise ValueError("Empty model inventory")
    identity = {name: value["sha256"] for name, value in files.items()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"roots": roots, "files": files, "fingerprint": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots", type=Path, help="JSON mapping of role names to installed directories"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    atomic_json(args.output, fingerprint(json.loads(args.roots.read_text())))
