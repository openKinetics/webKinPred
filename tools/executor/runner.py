from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from api.execution.artifacts import atomic_json, safe_relative, sha256_file


def configure(request):
    media = Path(os.environ.get("WEBKINPRED_MEDIA_ROOT", "/app/media"))
    media.mkdir(parents=True, exist_ok=True)
    work = Path("/work")
    mapping = request["payload"].get("seqmap", {})
    mapping.update({seq: sid for sid, seq in request["payload"].get("seq_id_to_seq", {}).items()})
    atomic_json(work / "seqmap.json", mapping)
    atomic_json(work / "molecular.json", request["payload"].get("molecular", {}))
    os.environ["SEQMAP_MANIFEST_PATH"] = str(work / "seqmap.json")
    os.environ["WEBKINPRED_MOLECULAR_MANIFEST"] = str(work / "molecular.json")
    os.environ["DJANGO_SETTINGS_MODULE"] = "webKinPred.settings_execution_runner"
    os.environ["WEBKINPRED_PROGRESS_FILE"] = "/work/progress.json"
    os.environ["KINFORM_MEDIA_PATH"] = str(media)
    os.environ["KINFORM_TOOLS_PATH"] = "/app/tools"
    os.environ["KINFORM_DATA"] = "/app/models/KinForm/results"
    os.environ["GPU_EMBED_REPO_ROOT"] = "/app"
    os.environ["GPU_EMBED_CACHE_SPOOL_DIR"] = "/work/tmp/cache"
    os.environ["GPU_EMBED_CACHE_SPOOL_FALLBACK_DIR"] = "/work/tmp/cache"
    os.environ["EITLEM_DELETE_EMBEDDINGS_AFTER_RUN"] = "0"
    os.environ["OMNIESI_DELETE_EMBEDDINGS_AFTER_RUN"] = "0"
    os.environ["WEBKINPRED_EMBED_WORKER"] = "1" if request["kind"] == "embedding" else "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MMSEQS_TMP_DIR"] = "/work/tmp/mmseqs"
    os.environ["CATPRED_CACHE_PATH"] = "/work/tmp/catpred"
    os.environ["DGLBACKEND"] = "pytorch"
    os.environ["DGLDEFAULTDIR"] = "/work/tmp/dgl"
    from api.prediction_engines.runtime_paths import DATA_PATHS, PYTHON_PATHS

    PYTHON_PATHS.update(request["settings"].get("python_paths", {}))
    DATA_PATHS["OmniESI-weights"] = "/work/tmp/omniesi"
    # Explicit validated model roots override legacy descriptor defaults too.
    checkpoint_root = (
        request["settings"].get("model_environment", {}).get("CATPRED_CHECKPOINT_ROOT")
    )
    if checkpoint_root:
        DATA_PATHS["CatPred_production_checkpoints"] = checkpoint_root
    # Only paths and bounded numerical knobs from validated profiles enter model environments.
    for key, value in request["settings"].get("model_environment", {}).items():
        if key.startswith(("KINFORM_", "CATPRED_", "CATRANGE_", "HF_", "TORCH_")):
            os.environ[key] = str(value)
    return media


def predict(request, media):
    import django

    django.setup()
    import pandas as pd
    from api.execution.context import StageContext, execution_job, stage_context
    from api.methods.registry import get
    from api.services.prediction_batch_service import build_sequence_batch_plan
    from api.tasks import _execute_target_batch

    spec = request["specification"]
    payload = request["payload"]
    frame = pd.read_csv(media / "jobs" / request["public_id"] / "input.csv")
    descriptors = [get(spec["methods"][t]) for t in spec["targets"]]
    plan = build_sequence_batch_plan(frame, descriptors, spec["handle_long_sequences"])

    def select_units(indices):
        chosen = set(indices)
        atomic_json(
            Path("/work/molecular-selected.json"),
            {
                method: [row for row in rows if row["unit_index"] in chosen]
                for method, rows in payload.get("molecular", {}).items()
            },
        )
        os.environ["WEBKINPRED_MOLECULAR_MANIFEST"] = "/work/molecular-selected.json"

    context = StageContext(
        request["public_id"],
        cache=payload.get("cache_snapshot", {}),
        cache_namespace=payload.get("cache_namespace", ""),
        unit_context=payload.get("unit_context", {}),
        select_units=select_units,
        unit_batch_size=max(
            1,
            int(
                request["settings"].get(
                    "unit_batch_size", request["settings"].get("batch_size", 128)
                )
            ),
        ),
    )
    experimental = []
    for row in payload.get("experimental_results", []):
        index = row.get("reaction_idx", row.get("idx"))
        if isinstance(index, int) and payload["offset"] <= index < payload["offset"] + len(frame):
            row = dict(row)
            for key in ["reaction_idx", "idx"]:
                if key in row:
                    row[key] -= payload["offset"]
            experimental.append(row)
    with stage_context(context):
        result = _execute_target_batch(
            job=execution_job(request["public_id"]),
            desc=get(request["method"]),
            df=frame,
            target=request["target"],
            sequences=list(plan.original_sequences),
            processed_by_reaction=list(plan.processed_by_reaction),
            valid_reaction_indices=list(plan.valid_reaction_indices),
            sequence_expansion=plan.expansion,
            experimental_results=experimental,
            canonicalize_substrates=spec["canonicalize_substrates"],
            disable_gpu_precompute=False,
            extra_call_kwargs={
                "cleanup_esm1v_embeddings": False,
                "cleanup_embeddings_after_run": False,
            },
            recon_xkg=spec.get("recon_xkg", False),
        )
    for index, reason in plan.skipped_reactions.items():
        result["preds"][index] = ""
        result["sources"][index] = reason
        result["extra"][index] = ""
    path = media / payload["outputs"][0]["path"]
    atomic_json(
        path,
        {
            "target": request["target"],
            "chunk": payload["chunk"],
            "offset": payload["offset"],
            "result": result,
            "cache_writes": context.cache_writes,
        },
    )
    return {"cache_writes": context.cache_writes}


def embedding(request, media):
    payload = request["payload"]
    if payload["step"] == "molecular":
        interpreters = {
            "UniKP": "unikp",
            "KinForm": "kinform_env",
            "CataPro": "catapro_env",
            "CatRange": "catrange_env",
        }
        python = (
            request["settings"]
            .get("python_paths", {})
            .get(
                payload["molecular_method"],
                f"/opt/conda/envs/{interpreters[payload['molecular_method']]}/bin/python",
            )
        )
        subprocess.run([python, "-m", "tools.executor.molecular", "/work/request.json"], check=True)
    elif payload["step"].startswith("kinform_"):
        from tools.executor.kinform import run

        run(request, media)
    else:
        from api.prediction_engines.runtime_paths import PYTHON_PATHS

        from tools.gpu_embed_service import run_step

        env = {
            "KINFORM_ESM_PATH": PYTHON_PATHS["esm2"],
            "KINFORM_ESMC_PATH": PYTHON_PATHS["esmc"],
            "KINFORM_T5_PATH": PYTHON_PATHS["t5"],
            "KINFORM_PSEQ2SITES_PATH": PYTHON_PATHS["pseq2sites"],
            "EITLEM_EMBED_PYTHON": PYTHON_PATHS["EITLEM"],
            "OMNIESI_EMBED_PYTHON": PYTHON_PATHS["OmniESI"],
            "CATPRED_EMBED_PYTHON": PYTHON_PATHS["CatPred"],
            "IECATA_EMBED_PYTHON": PYTHON_PATHS["IECata"],
        }
        os.environ.update(env)
        os.environ.update(request["settings"].get("embedding_environment", {}))
        step = payload["step"]
        batch_variable = {
            "prot_t5_mean": "GPU_EMBED_PROT_T5_MEAN_BATCH_SIZE",
            "eitlem_esm1v": "EITLEM_GPU_ESM1V_BATCH_SIZE",
            "omniesi_esm2": "OMNIESI_GPU_ESM_MAX_BATCH",
            "iecata_prot_t5_residues": "IECATA_GPU_PROT_T5_BATCH_SIZE",
            "catpred_embed": "CATPRED_GPU_ESM_MAX_BATCH",
            "turnup_esm1b": "TURNUP_EMBED_BATCH_SIZE",
        }.get(step)
        if batch_variable:
            # Retry profiles must control the actual child process batch, not
            # merely a setting on its orchestration process.
            os.environ[batch_variable] = str(max(1, int(request["settings"].get("batch_size", 1))))
        if step == "catpred_embed":
            step = "catpred_embed_kcat" if request["target"] == "kcat" else "catpred_embed_km"
        path = Path("/work/sequences.json")
        atomic_json(path, payload["seq_id_to_seq"])
        command = [
            sys.executable,
            run_step.__file__,
            "--step",
            step,
            "--repo-root",
            "/app",
            "--media-path",
            str(media),
            "--tools-path",
            "/app/tools",
            "--seq-id-to-seq-file",
            str(path),
            "--job-id",
            request["id"],
        ]
        subprocess.run(command, check=True)


def assemble(request, media):
    import pandas as pd
    from api.services.catrange_reporting import (
        build_prediction_result_frame,
        completed_reaction_count,
    )

    payload = request["payload"]
    target_order = request["specification"]["targets"]
    results = {}
    for path in (media / "execution/results").glob("*.json"):
        data = json.loads(path.read_text())
        results.setdefault(data["chunk"], {})[data["target"]] = path
    output = media / payload["outputs"][0]["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    for number, path in enumerate(
        sorted((media / "execution/inputs").glob("*.csv"), key=lambda p: int(p.stem))
    ):
        frame = pd.read_csv(path)
        target_results = {
            target: json.loads(result_path.read_text())["result"]
            for target, result_path in results[int(path.stem)].items()
        }
        if set(target_results) != set(target_order):
            raise ValueError("Assembly is missing a target")
        completed += completed_reaction_count(target_order, target_results)
        build_prediction_result_frame(frame, target_order, target_results).to_csv(
            output, index=False, mode="w" if number == 0 else "a", header=number == 0
        )
    return {"completed_reactions": completed}


def similarity(request, media):
    import django

    django.setup()
    import pandas as pd
    from api.execution.context import StageContext, stage_context
    from api.services.similarity_service import append_kcat_similarity_columns_to_output_csv

    path = media / "jobs" / request["public_id"] / "output.csv"
    temporary = path.with_suffix(".enriched.csv")
    context = StageContext(request["public_id"])
    result_paths = [media / entry["path"] for entry in request["payload"]["result_chunks"]]
    selections = iter(selected_sequence_rows(result_paths))
    with stage_context(context):
        for index, frame in enumerate(pd.read_csv(path, chunksize=128)):
            chunk = Path("/work/tmp/similarity.csv")
            frame.to_csv(chunk, index=False)
            selected = [next(selections) for _ in range(len(frame))]
            append_kcat_similarity_columns_to_output_csv(
                str(chunk),
                request["method"],
                recon_xkg=request["specification"].get("recon_xkg", False),
                selected_sequences_by_row=selected,
            )
            pd.read_csv(chunk).to_csv(
                temporary, mode="w" if index == 0 else "a", header=index == 0, index=False
            )
    if next(selections, None) is not None:
        raise ValueError("Similarity selections exceed the assembled row count")
    os.replace(temporary, path)
    return {"similarity_writes": context.similarity_writes}


def selected_sequence_rows(paths):
    """Stream kcat selections in original order across variable-sized CPU chunks."""
    offset = 0
    for candidate in paths:
        data = json.loads(candidate.read_text())
        if data["target"] != "kcat" or data["offset"] != offset:
            raise ValueError("Similarity selections are missing or out of order")
        count = len(data["result"]["preds"])
        selected = data["result"].get("selected_sequences")
        if selected is None:
            selected = [""] * count
        if len(selected) != count:
            raise ValueError("Similarity selection count does not match prediction rows")
        yield from selected
        offset += count


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    media = configure(request)
    checkpoint = {}
    if request["kind"] == "predict" and request["payload"].get("step") == "kinform_pool":
        from tools.executor.kinform import pool

        pool(request, media)
    elif request["kind"] == "predict" and request["payload"].get("step") == "kinform_binding_merge":
        from tools.executor.kinform import merge_binding

        merge_binding(request, media)
    elif request["kind"] == "predict":
        checkpoint = predict(request, media)
    elif request["kind"] == "embedding":
        embedding(request, media)
    elif request["kind"] == "assemble":
        checkpoint = (
            request["payload"].get("assembly_checkpoint", {})
            if request["payload"].get("phase") == "publish"
            else assemble(request, media)
        )
    elif request["kind"] == "similarity":
        checkpoint = similarity(request, media)
    else:
        raise ValueError("Unimplemented execution stage: " + request["kind"])
    metadata = {}
    if request["kind"] == "embedding" or request["payload"].get("step") in {
        "kinform_pool",
        "kinform_binding_merge",
    }:
        has_tensors = any(
            output["path"].endswith(".pt") for output in request["payload"]["outputs"]
        )
        if has_tensors:
            from api.prediction_engines.runtime_paths import PYTHON_PATHS

            method = "OmniESI" if request["method"] == "OmniESI-O2DENet" else request["method"]
            python = PYTHON_PATHS[method]
        else:
            python = sys.executable
        subprocess.run(
            [
                python,
                "-m",
                "tools.executor.validate_artifacts",
                sys.argv[1],
                str(media),
                "/work/artifact-metadata.json",
            ],
            check=True,
        )
        metadata = json.loads(Path("/work/artifact-metadata.json").read_text())
    outputs = []
    for output in request["payload"].get("outputs", []):
        path = media / safe_relative(output["path"])
        if not path.is_file():
            raise ValueError("Execution did not produce every declared output")
        outputs.append(
            {
                "id": output["id"],
                "identity": output["identity"],
                "checksum": sha256_file(path),
                "size": path.stat().st_size,
                "metadata": metadata.get(output["id"], {}),
                "source": str(path.relative_to("/work"))
                if path.is_relative_to("/work")
                else "media/" + output["path"],
            }
        )
    atomic_json(Path("/work/outcome.json"), {"artifacts": outputs, "checkpoint": checkpoint})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        from api.execution.errors import classify

        atomic_json(
            Path("/work/error.json"),
            {"category": classify(error), "exception": type(error).__name__},
        )
        raise
