from __future__ import annotations

import json
import os
import pickle
import subprocess
from pathlib import Path


def run(request, media):
    payload = request["payload"]
    mapping = payload["seq_id_to_seq"]
    Path("/work/seq_ids.txt").write_text("\n".join(mapping) + "\n")
    Path("/work/id_to_seq.pkl").write_bytes(pickle.dumps(mapping, protocol=4))
    Path("/work/sequences.json").write_text(json.dumps(mapping))
    root = Path("/app/models/KinForm/code")
    common = [
        "--seq_file",
        "/work/seq_ids.txt",
        "--id_to_seq_file",
        "/work/id_to_seq.pkl",
        "--batch_size",
        str(request["settings"].get("batch_size", 1)),
    ]
    step = payload["step"]
    if step == "kinform_t5_residues":
        command = [
            "/opt/conda/envs/prot_t5/bin/python",
            str(root / "protein_embeddings/t5_embeddings.py"),
            *common,
            "--setting",
            "residue",
            "--layers",
            "19",
            "None",
        ]
    elif step in {"kinform_esm2_residues", "kinform_esmc_residues"}:
        esmc = step == "kinform_esmc_residues"
        command = [
            f"/opt/conda/envs/{'esmc' if esmc else 'esm'}/bin/python",
            str(root / "protein_embeddings/prot_embeddings.py"),
            *common,
            "--setting",
            "residue",
            "--models",
            "esmc" if esmc else "esm2",
            "--layers",
            *(["24", "32"] if esmc else ["26", "29"]),
        ]
    elif step == "kinform_pseq2sites":
        for sid in mapping:
            if not (media / "sequence_info/prot_t5_last/residue_vecs" / (sid + ".npy")).is_file():
                raise ValueError(
                    "Missing T5 dependency; refusing to wait with GPU memory allocated"
                )
        command = [
            "/opt/conda/envs/pseq2sites/bin/python",
            str(root / "pseq2sites/pseq2sites_stream_worker.py"),
            "--seq-id-to-seq-file",
            "/work/sequences.json",
            "--binding-sites-path",
            str(
                Path("/work/pseq-cohort.tsv")
                if len(payload["outputs"]) > 1
                else media / payload["outputs"][0]["path"]
            ),
            "--batch-size",
            str(request["settings"].get("batch_size", 1)),
        ]
    else:
        raise ValueError("Unadmitted compound KinForm execution is forbidden")
    env = dict(
        os.environ, KINFORM_REQUIRE_CUDA="1", KINFORM_MEDIA_PATH=str(media), PYTHONPATH="/app"
    )
    command[0] = request["settings"].get("python_paths", {}).get(step, command[0])
    subprocess.run(command, check=True, env=env)
    if step == "kinform_pseq2sites" and len(payload["outputs"]) > 1:
        split_binding_outputs(Path("/work/pseq-cohort.tsv"), payload, media)


def split_binding_outputs(combined, payload, media):
    """Publish the original immutable partitions after one bounded model load."""
    import csv

    with combined.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        columns = reader.fieldnames
        rows = {}
        for row in reader:
            if row["PDB"] in rows:
                raise ValueError("Duplicate binding-site row")
            rows[row["PDB"]] = row
    if set(rows) != set(payload["seq_id_to_seq"]):
        raise ValueError("Binding-site cohort coverage differs")
    for output in payload["outputs"]:
        wanted = output["identity"]["context"]["sequences"]
        if not wanted or len(set(wanted)) != len(wanted) or not set(wanted) <= rows.keys():
            raise ValueError("Invalid binding-site partition")
        path = media / output["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows[sid] for sid in wanted)
    combined.unlink()


def merge_binding(request, media):
    import csv

    payload = request["payload"]
    destination = media / payload["outputs"][0]["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    header, seen = None, set()
    with destination.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        for part in payload["parts"]:
            with (media / part).open(newline="") as source:
                reader = csv.reader(source, delimiter="\t")
                columns = next(reader)
                if header is None:
                    header = columns
                    writer.writerow(header)
                elif columns != header:
                    raise ValueError("Binding-site partition headers differ")
                index = header.index("PDB")
                for row in reader:
                    if row[index] in seen:
                        raise ValueError("Duplicate binding-site partition row")
                    seen.add(row[index])
                    writer.writerow(row)
    if seen != set(payload["seq_id_to_seq"]):
        raise ValueError("Binding-site partitions do not cover the planned sequences")


def pool(request, media):
    import numpy as np

    from tools.gpu_embed_service.kinform_parallel_orchestrator import (
        _read_binding_site_scores,
        weighted_mean_from_residue,
    )

    scores = _read_binding_site_scores(media / "pseq2sites/binding_sites_all.tsv")
    for output in request["payload"]["outputs"]:
        path = media / output["path"]
        root = path.parent.parent
        sid = path.stem
        residue = np.load(root / "residue_vecs" / (sid + ".npy"), allow_pickle=False)
        value = (
            weighted_mean_from_residue(residue, scores[sid])
            if "weighted_vecs" in output["path"]
            else residue.mean(axis=0)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value.astype(np.float32))
