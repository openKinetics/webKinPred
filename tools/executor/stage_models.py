"""Stage model code without changing a dirty deployment or duplicating its weights.

The staged tree must be mounted read-only in attempts. Patches replace directory
entries atomically, so shared hardlinks never modify the original checkout.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from api.execution.artifacts import atomic_json, safe_relative, sha256_file


def stage(source, destination, patches):
    source = source.resolve()
    if destination.exists():
        raise ValueError("Use a new model release directory")
    shutil.copytree(source, destination, copy_function=os.link, symlinks=True)
    applied = {}
    for patch in sorted(patches.rglob("*")):
        if not patch.is_file():
            continue
        relative = safe_relative(str(patch.relative_to(patches)))
        target = destination / relative
        # Never follow a deployment symlink to overwrite another checkout.
        if any(parent.is_symlink() for parent in [target, *target.parents] if parent != Path("/")):
            raise ValueError("Model patch would follow a symlink: " + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with patch.open("rb") as incoming:
                shutil.copyfileobj(incoming, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, patch.stat().st_mode & 0o777)
        os.replace(temporary, target)
        applied[str(relative)] = sha256_file(target)
    manifest = {"source": str(source), "patches": applied, "mount_read_only": True}
    atomic_json(destination.parent / (destination.name + ".json"), manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("patches", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.destination, args.patches)))
