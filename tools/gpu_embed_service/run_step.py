#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


STEP_CHOICES = (
    "kinform_t5_full",
    "kinform_esm2_layers",
    "kinform_esmc_layers",
    "prot_t5_mean",
    "turnup_esm1b",
    "eitlem_esm1v",
    "catpred_embed_kcat",
    "catpred_embed_km",
    "omniesi_esm2",
    "realkcat_esm2_last_mean",
    "iecata_prot_t5_residues",
    "catrange_esmc",
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_in_home_env(env_name: str) -> str:
    return str((Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python").resolve())


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env)


def _parse_seq_ids(raw: str) -> list[str]:
    return [sid.strip() for sid in raw.split(",") if sid.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)


def _load_seq_id_to_seq(seq_ids: list[str], seqmap_db: Path) -> dict[str, str]:
    _ensure_exists(seqmap_db, "seqmap DB")
    if not seq_ids:
        return {}

    placeholders = ",".join(["?"] * len(seq_ids))
    sql = f"SELECT id, seq FROM sequences WHERE id IN ({placeholders})"
    # Use read-only URI mode to avoid WAL/lock file creation on NFS mounts.
    uri = f"file:{seqmap_db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        rows = con.execute(sql, seq_ids).fetchall()
    found = {str(row[0]): str(row[1]) for row in rows}
    missing = [sid for sid in seq_ids if sid not in found]
    if missing:
        raise RuntimeError(f"Missing seq IDs in seqmap DB: {','.join(missing)}")
    return {sid: found[sid] for sid in seq_ids}


def _write_temp_inputs(seq_id_to_seq: dict[str, str]) -> tuple[Path, Path, Path, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="gpu_embed_step_"))
    seq_file = tmp_dir / "seq_ids.txt"
    id_to_seq_pkl = tmp_dir / "id_to_seq.pkl"
    seq_map_json = tmp_dir / "seq_id_to_seq.json"

    with seq_file.open("w", encoding="utf-8") as handle:
        for seq_id in seq_id_to_seq.keys():
            handle.write(f"{seq_id}\n")

    with id_to_seq_pkl.open("wb") as handle:
        pickle.dump(seq_id_to_seq, handle, protocol=4)

    seq_map_json.write_text(json.dumps(seq_id_to_seq), encoding="utf-8")
    return tmp_dir, seq_file, id_to_seq_pkl, seq_map_json


def _kinform_env(repo_root: Path, media_path: Path, tools_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("KINFORM_MEDIA_PATH", str(media_path))
    env.setdefault("KINFORM_TOOLS_PATH", str(tools_path))
    env.setdefault("KINFORM_DATA", str((repo_root / "models" / "KinForm" / "results").resolve()))
    env.setdefault("KINFORM_ESM_PATH", _python_in_home_env("esm"))
    env.setdefault("KINFORM_ESMC_PATH", _python_in_home_env("esmc"))
    env.setdefault("KINFORM_T5_PATH", _python_in_home_env("prot_t5"))
    env.setdefault("KINFORM_PSEQ2SITES_PATH", _python_in_home_env("pseq2sites"))
    t5_model_default = (
        repo_root / "models" / "UniKP-main" / "models" / "protT5_xl" / "prot_t5_xl_uniref50"
    ).resolve()
    if t5_model_default.exists():
        env.setdefault("KINFORM_T5_MODEL_PATH", str(t5_model_default))
    # GPU embed service should run embedding workloads on CUDA, not CPU fallback.
    env.setdefault("KINFORM_REQUIRE_CUDA", "1")
    env.setdefault("GPU_REPO_ROOT", str(repo_root))
    repo_root_str = str(repo_root)
    existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
    if existing_pythonpath:
        parts = [p for p in existing_pythonpath.split(os.pathsep) if p]
        if repo_root_str not in parts:
            env["PYTHONPATH"] = os.pathsep.join([repo_root_str] + parts)
    else:
        env["PYTHONPATH"] = repo_root_str
    return env


def _run_kinform_pseq2sites(env: dict[str, str], seq_map_json: Path) -> None:
    py = env["KINFORM_PSEQ2SITES_PATH"]
    code = r"""
import json
import os
import sys
from pathlib import Path
import pandas as pd

repo_root = Path(os.environ["GPU_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root / "models" / "KinForm" / "code"))

from pseq2sites.get_sites import get_sites
from config import BS_PRED_PATH

seq_map = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bs_path = Path(BS_PRED_PATH)
bs_path.parent.mkdir(parents=True, exist_ok=True)
if bs_path.exists():
    bs_df = pd.read_csv(bs_path, sep="\t")
else:
    bs_df = pd.DataFrame(columns=["PDB", "Pred_BS_Scores"])
get_sites(seq_map, bs_df, save_path=str(bs_path), return_prot_t5=False)
"""
    _run([py, "-c", code, str(seq_map_json)], env)


def _run_kinform_esm2(env: dict[str, str], seq_file: Path, id_to_seq_pkl: Path) -> None:
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "prot_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    _run(
        [
            env["KINFORM_ESM_PATH"],
            str(script),
            "--seq_file",
            str(seq_file),
            "--models",
            "esm2",
            "--setting",
            "mean+weighted",
            "--weights_file",
            str(bs_pred),
            "--id_to_seq_file",
            str(id_to_seq_pkl),
            "--batch_size",
            "1",
        ],
        env,
    )


def _run_kinform_esmc(env: dict[str, str], seq_file: Path, id_to_seq_pkl: Path) -> None:
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "prot_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    _run(
        [
            env["KINFORM_ESMC_PATH"],
            str(script),
            "--seq_file",
            str(seq_file),
            "--models",
            "esmc",
            "--setting",
            "mean+weighted",
            "--weights_file",
            str(bs_pred),
            "--id_to_seq_file",
            str(id_to_seq_pkl),
            "--batch_size",
            "1",
        ],
        env,
    )


def _run_kinform_t5(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    *,
    mean_only: bool,
) -> None:
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "t5_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    cmd = [
        env["KINFORM_T5_PATH"],
        str(script),
        "--seq_file",
        str(seq_file),
        "--id_to_seq_file",
        str(id_to_seq_pkl),
        "--batch_size",
        "1",
    ]
    if mean_only:
        cmd.extend(["--setting", "mean", "--layers", "None"])
    else:
        cmd.extend(
            [
                "--setting",
                "mean+weighted",
                "--weights_file",
                str(bs_pred),
                "--layers",
                "19",
                "None",
            ]
        )
    _run(cmd, env)


def _run_prot_t5_mean(env: dict[str, str], seq_map_json: Path) -> None:
    t5_python = env["KINFORM_T5_PATH"]
    worker_script = (
        Path(env["GPU_REPO_ROOT"]) / "tools" / "gpu_embed_service" / "prot_t5_mean_worker.py"
    ).resolve()
    _ensure_exists(worker_script, "prot_t5_mean_worker.py")

    cache_dir = (Path(env["KINFORM_MEDIA_PATH"]) / "sequence_info" / "prot_t5_last" / "mean_vecs").resolve()
    batch_size = _env_int("GPU_EMBED_PROT_T5_MEAN_BATCH_SIZE", 8)
    async_workers = _env_int("GPU_EMBED_CACHE_ASYNC_WORKERS", 8)

    _run(
        [
            t5_python,
            str(worker_script),
            "--seq-id-to-seq-file",
            str(seq_map_json),
            "--cache-dir",
            str(cache_dir),
            "--batch-size",
            str(batch_size),
            "--async-workers",
            str(async_workers),
        ],
        env,
    )


def _run_kinform_t5_full_legacy_only(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    seq_map_json: Path,
) -> None:
    """Single T5 load path with early progress-visible cache writes.

    Flow:
    1) residue+mean for layers 19 + last (single model load)
    2) binding-site prediction from saved last-layer residue
    3) weighted vectors derived from residue files (no T5 reload)
    """
    script = (
        Path(env["GPU_REPO_ROOT"]) / "models" / "KinForm" / "code" / "protein_embeddings" / "t5_embeddings.py"
    ).resolve()
    bs_pred = (Path(env["KINFORM_MEDIA_PATH"]) / "pseq2sites" / "binding_sites_all.tsv").resolve()
    base_cmd = [
        env["KINFORM_T5_PATH"],
        str(script),
        "--seq_file", str(seq_file),
        "--id_to_seq_file", str(id_to_seq_pkl),
        "--batch_size", "1",
    ]

    # Phase 1: extract residue + mean for layers 19 + last in one model load.
    # Writing mean_vecs here lets embedding_progress track real progress while
    # weighted vectors wait for binding-site prediction.
    _run(base_cmd + ["--setting", "residue+mean", "--layers", "19", "None"], env)

    # Phase 2: predict binding sites — last-layer residue is already on disk, T5 is a no-op
    _run_kinform_pseq2sites(env, seq_map_json)

    # Phase 3: derive weighted from saved residue (Smart Reuse, no T5 reload).
    _run(
        base_cmd + [
            "--setting", "weighted",
            "--weights_file", str(bs_pred),
            "--layers", "19", "None",
        ],
        env,
    )


def _run_kinform_t5_full_legacy_with_esm(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    seq_map_json: Path,
) -> None:
    _run_kinform_t5_full_legacy_only(env, seq_file, id_to_seq_pkl, seq_map_json)
    _run_kinform_esm2(env, seq_file, id_to_seq_pkl)
    _run_kinform_esmc(env, seq_file, id_to_seq_pkl)


def _run_kinform_t5_full(
    env: dict[str, str],
    seq_file: Path,
    id_to_seq_pkl: Path,
    seq_map_json: Path,
    *,
    seq_id_to_seq: dict[str, str],
    job_id: str | None = None,
) -> None:
    parallel_enabled = _env_bool("KINFORM_PARALLEL_EMBED_ENABLE", True)
    allow_fallback = _env_bool("KINFORM_PARALLEL_EMBED_ALLOW_LEGACY_FALLBACK", True)

    if not parallel_enabled:
        _run_kinform_t5_full_legacy_only(env, seq_file, id_to_seq_pkl, seq_map_json)
        return

    orchestrator_script = (
        Path(env["GPU_REPO_ROOT"]) / "tools" / "gpu_embed_service" / "kinform_parallel_orchestrator.py"
    ).resolve()
    _ensure_exists(orchestrator_script, "kinform parallel orchestrator script")

    try:
        _run(
            [
                env["KINFORM_T5_PATH"],
                str(orchestrator_script),
                "--seq-id-to-seq-file",
                str(seq_map_json),
                "--repo-root",
                str(Path(env["GPU_REPO_ROOT"]).resolve()),
                "--media-path",
                str(Path(env["KINFORM_MEDIA_PATH"]).resolve()),
                "--job-id",
                str(job_id or ""),
            ],
            env,
        )
        return
    except Exception as exc:
        if not allow_fallback:
            raise
        print(
            "KINFORM_PARALLEL_FALLBACK "
            f"job_id={job_id or 'unknown'} "
            f"reason={exc.__class__.__name__}:{exc}"
        )
        _run_kinform_t5_full_legacy_with_esm(env, seq_file, id_to_seq_pkl, seq_map_json)


def _run_turnup(env: dict[str, str], seq_map_json: Path) -> None:
    turnup_env = dict(env)
    turnup_env.setdefault("TURNUP_MEDIA_PATH", env["KINFORM_MEDIA_PATH"])
    turnup_env.setdefault("TURNUP_TOOLS_PATH", env["KINFORM_TOOLS_PATH"])
    turnup_env.setdefault(
        "TURNUP_DATA_PATH",
        str((Path(env["GPU_REPO_ROOT"]) / "models" / "TurNup" / "data").resolve()),
    )
    turnup_python = (
        os.environ.get("TURNUP_EMBED_PYTHON")
        or os.environ.get("KINFORM_ESM_PATH")
        or _python_in_home_env("esm")
    )
    code = r"""
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["GPU_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root / "models" / "TurNup" / "code"))
from enzyme_representations import calcualte_esm1b_ts_vectors

seq_map = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
seqs = list(dict.fromkeys(seq_map.values()))
calcualte_esm1b_ts_vectors(seqs)
"""
    _run([turnup_python, "-c", code, str(seq_map_json)], turnup_env)


def _run_eitlem_esm1v(env: dict[str, str], seq_map_json: Path) -> None:
    eitlem_python = (
        os.environ.get("EITLEM_EMBED_PYTHON")
        or os.environ.get("EITLEM_PYTHON")
        or sys.executable
    )
    worker_script = (
        Path(env["GPU_REPO_ROOT"]) / "tools" / "gpu_embed_service" / "eitlem_esm1v_worker.py"
    ).resolve()
    _ensure_exists(worker_script, "eitlem_esm1v_worker.py")

    cache_dir = (Path(env["KINFORM_MEDIA_PATH"]) / "sequence_info" / "esm1v").resolve()
    batch_size = _env_int("EITLEM_GPU_ESM1V_BATCH_SIZE", 4)
    async_workers = _env_int("GPU_EMBED_CACHE_ASYNC_WORKERS", 8)
    cmd = [
        eitlem_python,
        str(worker_script),
        "--seq-id-to-seq-file",
        str(seq_map_json),
        "--cache-dir",
        str(cache_dir),
        "--batch-size",
        str(batch_size),
        "--async-workers",
        str(async_workers),
    ]
    model_path = str(os.environ.get("EITLEM_MODEL_PATH", "")).strip()
    if model_path:
        cmd.extend(["--model-path", model_path])
    _run(cmd, env)


def _run_catpred_embed(env: dict[str, str], seq_map_json: Path, *, parameter: str) -> None:
    catpred_python = (
        os.environ.get("CATPRED_EMBED_PYTHON")
        or os.environ.get("CATPRED_PYTHON")
        or sys.executable
    )
    worker_script = (
        Path(env["GPU_REPO_ROOT"])
        / "models"
        / "CatPred"
        / "catpred"
        / "integration"
        / "catpred_embed_gpu.py"
    ).resolve()
    _ensure_exists(worker_script, "catpred_embed_gpu.py")

    checkpoint_root = Path(
        os.environ.get(
            "CATPRED_CHECKPOINT_ROOT",
            str(
                (
                    Path(env["GPU_REPO_ROOT"])
                    / "models"
                    / "CatPred"
                    / "data"
                    / "pretrained"
                    / "production"
                ).resolve()
            ),
        )
    ).resolve()
    cache_root = (Path(env["KINFORM_MEDIA_PATH"]) / "sequence_info" / "catpred_esm2").resolve()
    # CatPred secure deserialization requires checkpoint paths to be under trusted roots.
    # Extend (do not replace) any existing trust roots with the configured checkpoint root.
    trusted_roots_var = "CATPRED_TRUSTED_DESERIALIZATION_ROOTS"
    existing_roots = [p for p in str(env.get(trusted_roots_var, "")).split(os.pathsep) if p.strip()]
    checkpoint_root_str = str(checkpoint_root)
    if checkpoint_root_str not in existing_roots:
        existing_roots.append(checkpoint_root_str)
    env[trusted_roots_var] = os.pathsep.join(existing_roots)

    _run(
        [
            catpred_python,
            str(worker_script),
            "--seq-id-to-seq-file",
            str(seq_map_json),
            "--parameter",
            parameter,
            "--checkpoint-root",
            str(checkpoint_root),
            "--cache-root",
            str(cache_root),
        ],
        env,
    )


def _run_omniesi_esm2(env: dict[str, str], seq_map_json: Path) -> None:
    omniesi_python = (
        os.environ.get("OMNIESI_EMBED_PYTHON")
        or os.environ.get("OmniESI_EMBED_PYTHON")
        or os.environ.get("OMNIESI_PYTHON")
        or sys.executable
    )
    worker_script = (
        Path(env["GPU_REPO_ROOT"]) / "tools" / "gpu_embed_service" / "omniesi_esm2_worker.py"
    ).resolve()
    _ensure_exists(worker_script, "omniesi_esm2_worker.py")

    cache_dir = (Path(env["KINFORM_MEDIA_PATH"]) / "sequence_info" / "omniesi_esm2").resolve()
    token_budget = _env_int("OMNIESI_GPU_ESM_TOKEN_BUDGET", 6000)
    max_batch = _env_int("OMNIESI_GPU_ESM_MAX_BATCH", 8)
    async_workers = _env_int("GPU_EMBED_CACHE_ASYNC_WORKERS", 8)

    _run(
        [
            omniesi_python,
            str(worker_script),
            "--seq-id-to-seq-file",
            str(seq_map_json),
            "--cache-dir",
            str(cache_dir),
            "--token-budget",
            str(token_budget),
            "--max-batch",
            str(max_batch),
            "--async-workers",
            str(async_workers),
        ],
        env,
    )


def _run_realkcat_esm2_last_mean(env: dict[str, str], seq_map_json: Path) -> None:
    realkcat_python = (
        os.environ.get("REALKCAT_EMBED_PYTHON")
        or os.environ.get("REALKCAT_PYTHON")
        or os.environ.get("OMNIESI_EMBED_PYTHON")
        or os.environ.get("OMNIESI_PYTHON")
        or sys.executable
    )
    worker_script = (
        Path(env["GPU_REPO_ROOT"]) / "tools" / "gpu_embed_service" / "realkcat_esm2_last_mean_worker.py"
    ).resolve()
    _ensure_exists(worker_script, "realkcat_esm2_last_mean_worker.py")

    cache_dir = (Path(env["KINFORM_MEDIA_PATH"]) / "sequence_info" / "esm2_layer_last_mean").resolve()
    token_budget = _env_int("REALKCAT_GPU_ESM_TOKEN_BUDGET", 6000)
    max_batch = _env_int("REALKCAT_GPU_ESM_MAX_BATCH", 8)
    async_workers = _env_int("GPU_EMBED_CACHE_ASYNC_WORKERS", 8)

    _run(
        [
            realkcat_python,
            str(worker_script),
            "--seq-id-to-seq-file",
            str(seq_map_json),
            "--cache-dir",
            str(cache_dir),
            "--token-budget",
            str(token_budget),
            "--max-batch",
            str(max_batch),
            "--async-workers",
            str(async_workers),
        ],
        env,
    )


def _run_iecata_prot_t5_residues(env: dict[str, str], seq_map_json: Path) -> None:
    iecata_python = (
        os.environ.get("IECATA_EMBED_PYTHON")
        or os.environ.get("IECATA_PYTHON")
        or env.get("KINFORM_T5_PATH")          # reuse the T5 env if no dedicated one
        or sys.executable
    )
    worker_script = (
        Path(env["GPU_REPO_ROOT"]) / "tools" / "gpu_embed_service" / "iecata_prot_t5_residues_worker.py"
    ).resolve()
    _ensure_exists(worker_script, "iecata_prot_t5_residues_worker.py")

    cache_dir = (
        Path(env["KINFORM_MEDIA_PATH"]) / "sequence_info" / "iecata_prot_t5_residues"
    ).resolve()
    batch_size = _env_int("IECATA_GPU_PROT_T5_BATCH_SIZE", 4)
    async_workers = _env_int("GPU_EMBED_CACHE_ASYNC_WORKERS", 8)

    _run(
        [
            iecata_python,
            str(worker_script),
            "--seq-id-to-seq-file",
            str(seq_map_json),
            "--cache-dir",
            str(cache_dir),
            "--batch-size",
            str(batch_size),
            "--async-workers",
            str(async_workers),
        ],
        env,
    )

def _run_catrange_esmc(env: dict[str, str], seq_map_json: Path) -> None:
    esmc_python = (
        os.environ.get("CATRANGE_EMBED_PYTHON")
        or env.get("KINFORM_ESMC_PATH")
        or _python_in_home_env("esmc")
    )
    worker_script = (
        Path(env["GPU_REPO_ROOT"]) / "tools" / "gpu_embed_service" / "catrange_esmc_worker.py"
    ).resolve()
    _ensure_exists(worker_script, "catrange_esmc_worker.py")

    cache_dir = (Path(env["KINFORM_MEDIA_PATH"]) / "sequence_info" / "catrange_esmc").resolve()
    async_workers = _env_int("GPU_EMBED_CACHE_ASYNC_WORKERS", 8)

    _run(
        [
            esmc_python,
            str(worker_script),
            "--seq-id-to-seq-file",
            str(seq_map_json),
            "--cache-dir",
            str(cache_dir),
            "--async-workers",
            str(async_workers),
        ],
        env,
    )


def run_step(
    step: str,
    seq_ids: list[str],
    repo_root: Path,
    media_path: Path,
    tools_path: Path,
    *,
    seq_id_to_seq: dict[str, str] | None = None,
    job_id: str | None = None,
) -> None:
    if not seq_ids:
        print("No sequence IDs provided; nothing to do.")
        return

    if seq_id_to_seq is None:
        seqmap_db = (media_path / "sequence_info" / "seqmap.sqlite3").resolve()
        seq_id_to_seq = _load_seq_id_to_seq(seq_ids, seqmap_db)
    else:
        missing = [sid for sid in seq_ids if sid not in seq_id_to_seq]
        if missing:
            raise RuntimeError(f"Missing seq IDs in provided map: {','.join(missing)}")

    tmp_dir, seq_file, id_to_seq_pkl, seq_map_json = _write_temp_inputs(seq_id_to_seq)
    env = _kinform_env(repo_root=repo_root, media_path=media_path, tools_path=tools_path)

    try:
        if step == "kinform_t5_full":
            _run_kinform_t5_full(
                env,
                seq_file,
                id_to_seq_pkl,
                seq_map_json,
                seq_id_to_seq=seq_id_to_seq,
                job_id=job_id,
            )
        elif step == "kinform_esm2_layers":
            _run_kinform_esm2(env, seq_file, id_to_seq_pkl)
        elif step == "kinform_esmc_layers":
            _run_kinform_esmc(env, seq_file, id_to_seq_pkl)
        elif step == "prot_t5_mean":
            _run_prot_t5_mean(env, seq_map_json)
        elif step == "turnup_esm1b":
            _run_turnup(env, seq_map_json)
        elif step == "eitlem_esm1v":
            _run_eitlem_esm1v(env, seq_map_json)
        elif step == "catpred_embed_kcat":
            _run_catpred_embed(env, seq_map_json, parameter="kcat")
        elif step == "catpred_embed_km":
            _run_catpred_embed(env, seq_map_json, parameter="km")
        elif step == "omniesi_esm2":
            _run_omniesi_esm2(env, seq_map_json)
        elif step == "realkcat_esm2_last_mean":
            _run_realkcat_esm2_last_mean(env, seq_map_json)
        elif step == "iecata_prot_t5_residues":
            _run_iecata_prot_t5_residues(env, seq_map_json)
        elif step == "catrange_esmc":
            _run_catrange_esmc(env, seq_map_json)
        else:
            raise RuntimeError(f"Unsupported step: {step}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one GPU embedding step for a seq-id batch.")
    parser.add_argument("--step", required=True, choices=STEP_CHOICES)
    parser.add_argument("--seq-ids", default="", help="Comma-separated sequence IDs (omit when --seq-id-to-seq-file is provided)")
    parser.add_argument("--repo-root", default=os.environ.get("GPU_EMBED_REPO_ROOT", ""))
    parser.add_argument("--media-path", default=os.environ.get("KINFORM_MEDIA_PATH", ""))
    parser.add_argument("--tools-path", default=os.environ.get("KINFORM_TOOLS_PATH", ""))
    parser.add_argument("--job-id", default=os.environ.get("GPU_EMBED_JOB_ID", ""))
    parser.add_argument(
        "--seq-id-to-seq-json",
        default="",
        help="JSON string mapping seq_id→sequence (skips seqmap DB lookup)",
    )
    parser.add_argument(
        "--seq-id-to-seq-file",
        default="",
        help="Path to JSON file mapping seq_id→sequence (skips seqmap DB lookup)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _default_repo_root().resolve()
    media_path = Path(args.media_path).resolve() if args.media_path else Path("/mnt/webkinpred/media").resolve()
    tools_path = Path(args.tools_path).resolve() if args.tools_path else (repo_root / "tools").resolve()
    seq_id_to_seq: dict[str, str] | None = None
    if args.seq_id_to_seq_file:
        seq_id_to_seq = json.loads(Path(args.seq_id_to_seq_file).read_text(encoding="utf-8"))
        seq_ids = _parse_seq_ids(args.seq_ids) if args.seq_ids else list(seq_id_to_seq.keys())
    elif args.seq_id_to_seq_json:
        seq_id_to_seq = json.loads(args.seq_id_to_seq_json)
        seq_ids = _parse_seq_ids(args.seq_ids)
    else:
        seq_ids = _parse_seq_ids(args.seq_ids)

    run_step(
        step=args.step,
        seq_ids=seq_ids,
        repo_root=repo_root,
        media_path=media_path,
        tools_path=tools_path,
        seq_id_to_seq=seq_id_to_seq,
        job_id=(args.job_id or None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
