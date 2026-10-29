"""Recover only complete, validated files from an identical immutable stage."""

import csv
import json
import shutil
import zipfile

import numpy as np
from api.execution.artifacts import atomic_json, safe_relative, sha256_file


def valid_array(path, method):
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if not values.size:
            return False
        if values.dtype.kind == "U":
            return method == "molecular-CatRange" and values.shape == (1,)
        if values.dtype.kind != "f":
            return False
        flat = values.reshape(-1)
        return all(np.isfinite(flat[i : i + 1024**2]).all() for i in range(0, flat.size, 1024**2))
    except (OSError, ValueError, EOFError):
        return False


def complete_tensor_archive(path):
    # Tensor contents are checked by the model interpreter before final commit.
    # Here, preserve complete atomic torch.save files without importing Torch
    # (and its model-sized dependencies) into the trusted executor agent.
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return (
                any(name.endswith("/data.pkl") for name in names)
                and any("/data/" in name for name in names)
                and archive.testzip() is None
            )
    except (OSError, ValueError, zipfile.BadZipFile, EOFError):
        return False


def complete_output(path, method):
    return (path.suffix == ".npy" and valid_array(path, method)) or (
        path.suffix == ".pt" and complete_tensor_archive(path)
    )


def partial_manifest(request, work):
    outputs = []
    if request["kind"] != "embedding":
        return {"artifacts": outputs}
    for output in request["payload"].get("outputs", []):
        path = work / "media" / safe_relative(output["path"])
        if not complete_output(path, request["method"]):
            continue
        outputs.append(
            {
                "id": output["id"],
                "identity": output["identity"],
                "checksum": sha256_file(path),
                "size": path.stat().st_size,
                "source": str(path.relative_to(work)),
            }
        )
    return {"artifacts": outputs}


def remember(root, request):
    index = root / "checkpoints" / (request["stage_id"] + ".json")
    if index.exists():
        previous = json.loads(index.read_text())["attempt"]
        if previous == request["id"]:
            return
        prior = root / "attempts" / safe_relative(previous) / "request.json"
        current = root / "attempts" / safe_relative(request["id"]) / "request.json"
        if prior.exists() and prior.stat().st_mtime_ns > current.stat().st_mtime_ns:
            return
    atomic_json(index, {"attempt": request["id"]})


def restore(root, request, work):
    index = root / "checkpoints" / (request["stage_id"] + ".json")
    if request["kind"] != "embedding" or not index.exists():
        return
    previous = root / "attempts" / safe_relative(json.loads(index.read_text())["attempt"])
    old = json.loads((previous / "request.json").read_text())
    if old["stage_id"] != request["stage_id"] or old["specification"] != request["specification"]:
        return
    allowed = {
        (value["id"], value["path"]): value["identity"]
        for value in old["payload"].get("outputs", [])
    }
    for output in request["payload"].get("outputs", []):
        if allowed.get((output["id"], output["path"])) != output["identity"]:
            continue
        relative = safe_relative(output["path"])
        source, destination = previous / "media" / relative, work / "media" / relative
        if (
            not destination.exists()
            and source.is_file()
            and source.suffix == ".tsv"
            and request["payload"].get("step") == "kinform_pseq2sites"
        ):
            preserved = []
            with source.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    sequence = request["payload"].get("seq_id_to_seq", {}).get(row.get("PDB"))
                    if not sequence or not row.get("Pred_BS_Scores"):
                        continue
                    try:
                        values = np.asarray(
                            [float(value) for value in row["Pred_BS_Scores"].split(",")]
                        )
                    except ValueError:
                        continue
                    if (
                        len(values) == min(1024, len(sequence.replace(",", "")))
                        and np.isfinite(values).all()
                    ):
                        preserved.append([row["PDB"], row["Pred_BS_Scores"]])
            if preserved:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("w", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerow(["PDB", "Pred_BS_Scores"])
                    writer.writerows(preserved)
            continue
        if destination.exists() or not complete_output(source, request["method"]):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
