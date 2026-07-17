from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from api.prediction_engines.runtime_paths import DATA_PATHS
from tools.gpu_embed_service.cache_io import resolve_missing_ids


@dataclass(frozen=True)
class EmbeddingStepPlan:
    step_key: str
    missing_seq_ids: list[str]
    required_paths_by_seq: dict[str, set[str]]
    gpu_enabled: bool = True


@dataclass(frozen=True)
class EmbeddingPlan:
    method_key: str
    target: str
    profile: str | None
    gpu_supported: bool
    gpu_reason: str | None
    media_path: Path
    tools_path: Path
    sequences: list[str]
    seq_ids: list[str]
    seq_id_to_seq: dict[str, str]
    expected_paths_by_seq: dict[str, set[str]]
    missing_paths_by_seq: dict[str, set[str]]
    path_to_seqs: dict[str, set[str]]
    watch_dirs: set[Path]
    total: int
    cached_already: int
    need_computation: int
    step_plans: list[EmbeddingStepPlan]


def method_env_keys(method_key: str) -> tuple[str | None, str | None]:
    mapping = {
        "UniKP": ("UNIKP_MEDIA_PATH", "UNIKP_TOOLS_PATH"),
        "EITLEM": ("EITLEM_MEDIA_PATH", "EITLEM_TOOLS_PATH"),
        "TurNup": ("TURNUP_MEDIA_PATH", "TURNUP_TOOLS_PATH"),
        "CataPro": ("CATAPRO_MEDIA_PATH", "CATAPRO_TOOLS_PATH"),
        "CatPred": ("CATPRED_MEDIA_PATH", "CATPRED_TOOLS_PATH"),
        "KinForm-H": ("KINFORM_MEDIA_PATH", "KINFORM_TOOLS_PATH"),
        "KinForm-L": ("KINFORM_MEDIA_PATH", "KINFORM_TOOLS_PATH"),
        "IECata": ("IECATA_MEDIA_PATH", "IECATA_TOOLS_PATH"),
        "RealKcat": ("REALKCAT_MEDIA_PATH", "REALKCAT_TOOLS_PATH"),
    }
    return mapping.get(method_key, (None, None))


def resolve_media_and_tools(method_key: str, env: dict) -> tuple[Path, Path]:
    media_key, tools_key = method_env_keys(method_key)

    media_path = env.get(media_key) if media_key else None
    tools_path = env.get(tools_key) if tools_key else None

    if not media_path:
        media_path = DATA_PATHS.get("media")
    if not tools_path:
        tools_path = DATA_PATHS.get("tools")

    if not media_path or not tools_path:
        raise RuntimeError("Could not resolve media/tools paths for embedding planning.")

    return Path(media_path).resolve(), Path(tools_path).resolve()


def normalise_sequences_for_method(method_key: str, sequences: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for raw in sequences:
        seq = str(raw).strip()
        if not seq:
            continue
        if method_key == "TurNup":
            # TurNup upper-cases and truncates to 1022 via preprocess_enzymes().
            seq = seq.upper()[:1022]
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out


def resolve_seq_ids_via_cli(sequences: list[str], tools_path: Path, media_path: Path) -> list[str]:
    seqmap_cli = tools_path / "seqmap" / "main.py"
    seqmap_db = media_path / "sequence_info" / "seqmap.sqlite3"
    if not seqmap_cli.exists():
        raise RuntimeError(f"seqmap CLI not found: {seqmap_cli}")
    if not seqmap_db.exists():
        raise RuntimeError(f"seqmap DB not found: {seqmap_db}")

    payload = "\n".join(sequences) + "\n"
    cmd = [
        sys.executable,
        str(seqmap_cli),
        "--db",
        str(seqmap_db),
        "batch-get-or-create",
        "--stdin",
    ]
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"seqmap failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    seq_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(seq_ids) != len(sequences):
        raise RuntimeError(f"seqmap returned {len(seq_ids)} ids for {len(sequences)} sequences")
    return seq_ids


def _catpred_parameter(target: str) -> str | None:
    if target == "kcat":
        return "kcat"
    if target in {"Km", "km"}:
        return "km"
    return None


def _catpred_checkpoint_key(model_pt_path: Path) -> str:
    parent = model_pt_path.parent.name
    grandparent = model_pt_path.parent.parent.name
    return f"{grandparent}__{parent}" if grandparent else parent


def _discover_catpred_checkpoint_keys(checkpoint_root: Path, parameter: str) -> list[str]:
    parameter_root = checkpoint_root / parameter
    if not parameter_root.exists():
        return []

    keys: list[str] = []
    seen: set[str] = set()
    for model_file in sorted(parameter_root.rglob("model.pt")):
        key = _catpred_checkpoint_key(model_file)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def expected_paths_by_seq(
    *,
    method_key: str,
    target: str,
    seq_ids: list[str],
    media_path: Path,
    env: dict,
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}

    if method_key in {"UniKP", "CataPro"}:
        base = media_path / "sequence_info" / "prot_t5_last" / "mean_vecs"
        for seq_id in seq_ids:
            out[seq_id] = {str((base / f"{seq_id}.npy").resolve())}
        return out

    if method_key == "EITLEM":
        # Full per-residue ESM1v matrices.  These are ephemeral: written by the
        # GPU step (or CPU fallback inside the prediction script) and deleted by
        # the prediction script after all predictions are complete.
        base = media_path / "sequence_info" / "esm1v"
        for seq_id in seq_ids:
            out[seq_id] = {str((base / f"{seq_id}.npy").resolve())}
        return out

    if method_key == "TurNup":
        base = media_path / "sequence_info" / "esm1b_turnup"
        for seq_id in seq_ids:
            out[seq_id] = {str((base / f"{seq_id}.npy").resolve())}
        return out

    if method_key in {"KinForm-H", "KinForm-L"}:
        base = media_path / "sequence_info"
        roots = [
            "esm2_layer_26",
            "esm2_layer_29",
            "esmc_layer_24",
            "esmc_layer_32",
            "prot_t5_layer_19",
            "prot_t5_last",
        ]
        for seq_id in seq_ids:
            paths: set[str] = set()
            for root in roots:
                paths.add(str((base / root / "mean_vecs" / f"{seq_id}.npy").resolve()))
                paths.add(str((base / root / "weighted_vecs" / f"{seq_id}.npy").resolve()))
            out[seq_id] = paths
        return out

    if method_key == "CatPred":
        parameter = _catpred_parameter(target)
        if parameter is None:
            return {}

        checkpoint_root = env.get("CATPRED_CHECKPOINT_ROOT") or DATA_PATHS.get(
            "CatPred_production_checkpoints"
        )
        if not checkpoint_root:
            return {}

        model_keys = _discover_catpred_checkpoint_keys(Path(checkpoint_root).resolve(), parameter)
        if not model_keys:
            return {}

        base = media_path / "sequence_info" / "catpred_esm2" / parameter
        for seq_id in seq_ids:
            out[seq_id] = {
                str((base / model_key / f"{seq_id}.pt").resolve()) for model_key in model_keys
            }
        return out

    if method_key in {"OmniESI", "OmniESI-O2DENet"}:
        base = media_path / "sequence_info" / "omniesi_esm2"
        for seq_id in seq_ids:
            out[seq_id] = {str((base / f"{seq_id}.pt").resolve())}
        return out

    if method_key == "RealKcat":
        base = media_path / "sequence_info" / "esm2_layer_last_mean"
        for seq_id in seq_ids:
            out[seq_id] = {str((base / f"{seq_id}.npy").resolve())}
        return out

    if method_key == "IECata":
        base = media_path / "sequence_info" / "iecata_prot_t5_residues"
        for seq_id in seq_ids:
            out[seq_id] = {str((base / f"{seq_id}.npy").resolve())}
        return out

    return {}


def _partition_missing(expected: dict[str, set[str]]) -> tuple[dict[str, set[str]], dict[str, set[str]], set[Path], int, int, int]:
    missing_paths_by_seq = missing_paths_by_seq_from_snapshot(expected)
    path_to_seqs: dict[str, set[str]] = {}
    watch_dirs: set[Path] = set()

    cached_already = 0
    need_computation = 0

    for seq_id, paths in expected.items():
        missing = missing_paths_by_seq.get(seq_id, set())
        cached_already += len(paths) - len(missing)
        need_computation += len(missing)

        if missing:
            missing_paths_by_seq[seq_id] = missing
            for path_str in missing:
                path_to_seqs.setdefault(path_str, set()).add(seq_id)
                watch_dirs.add(Path(path_str).parent)

    total = sum(len(paths) for paths in expected.values())
    return (
        missing_paths_by_seq,
        path_to_seqs,
        watch_dirs,
        total,
        cached_already,
        need_computation,
    )


def _step_from_paths(
    step_key: str,
    step_paths_by_seq: dict[str, set[str]],
    *,
    gpu_enabled: bool = True,
) -> EmbeddingStepPlan:
    missing = sorted(missing_paths_by_seq_from_snapshot(step_paths_by_seq).keys())
    return EmbeddingStepPlan(
        step_key=step_key,
        missing_seq_ids=missing,
        required_paths_by_seq=step_paths_by_seq,
        gpu_enabled=gpu_enabled,
    )


def _load_binding_site_ids(binding_sites_path: Path) -> set[str]:
    if not binding_sites_path.exists():
        return set()

    found: set[str] = set()
    try:
        with binding_sites_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames and "PDB" in reader.fieldnames:
                for row in reader:
                    seq_id = str(row.get("PDB", "")).strip()
                    if seq_id:
                        found.add(seq_id)
            else:
                # Fallback: try first column as sequence ID.
                handle.seek(0)
                simple = csv.reader(handle, delimiter="\t")
                for idx, row in enumerate(simple):
                    if idx == 0:
                        continue
                    if row:
                        seq_id = str(row[0]).strip()
                        if seq_id:
                            found.add(seq_id)
    except Exception:
        # Fail-open: if file is unreadable, we'll treat all seq IDs as missing.
        return set()
    return found


def _profile_for_method(method_key: str) -> tuple[str | None, bool, str | None]:
    if method_key in {"KinForm-H", "KinForm-L"}:
        return "kinform_full", True, None
    if method_key in {"CataPro", "UniKP"}:
        return "prot_t5_mean", True, None
    if method_key == "TurNup":
        return "turnup_esm1b", True, None
    if method_key == "EITLEM":
        return "eitlem_esm1v", True, None
    if method_key == "CatPred":
        return "catpred_embed", True, None
    if method_key in {"OmniESI", "OmniESI-O2DENet"}:
        return "omniesi_esm2", True, None
    if method_key == "RealKcat":
        return "realkcat_esm2_last_mean", True, None
    if method_key == "IECata":
        return "iecata_prot_t5_residues", True, None
    return None, False, "gpu_offload_not_applicable"


def _step_plans_for_profile(
    *,
    profile: str | None,
    method_key: str,
    target: str,
    seq_ids: list[str],
    media_path: Path,
    env: dict | None = None,
    full_expected_paths: dict[str, set[str]] | None = None,
) -> list[EmbeddingStepPlan]:
    if profile == "eitlem_esm1v":
        base = media_path / "sequence_info" / "esm1v"
        paths = {sid: {str((base / f"{sid}.npy").resolve())} for sid in seq_ids}
        return [_step_from_paths("eitlem_esm1v", paths)]

    if profile == "catpred_embed":
        parameter = _catpred_parameter(target)
        if parameter is None or not full_expected_paths:
            return []
        step_key = f"catpred_embed_{parameter}"
        return [_step_from_paths(step_key, full_expected_paths)]

    if profile == "prot_t5_mean":
        base = media_path / "sequence_info" / "prot_t5_last" / "mean_vecs"
        paths = {sid: {str((base / f"{sid}.npy").resolve())} for sid in seq_ids}
        return [_step_from_paths("prot_t5_mean", paths)]

    if profile == "turnup_esm1b":
        base = media_path / "sequence_info" / "esm1b_turnup"
        paths = {sid: {str((base / f"{sid}.npy").resolve())} for sid in seq_ids}
        return [_step_from_paths("turnup_esm1b", paths)]

    if profile == "omniesi_esm2":
        base = media_path / "sequence_info" / "omniesi_esm2"
        paths = {sid: {str((base / f"{sid}.pt").resolve())} for sid in seq_ids}
        return [_step_from_paths("omniesi_esm2", paths)]

    if profile == "realkcat_esm2_last_mean":
        base = media_path / "sequence_info" / "esm2_layer_last_mean"
        paths = {sid: {str((base / f"{sid}.npy").resolve())} for sid in seq_ids}
        return [_step_from_paths("realkcat_esm2_last_mean", paths)]

    if profile == "iecata_prot_t5_residues":
        base = media_path / "sequence_info" / "iecata_prot_t5_residues"
        paths = {sid: {str((base / f"{sid}.npy").resolve())} for sid in seq_ids}
        return [_step_from_paths("iecata_prot_t5_residues", paths)]

    if profile == "kinform_full":
        base = media_path / "sequence_info"
        esm2_paths = {
            sid: {
                str((base / "esm2_layer_26" / "mean_vecs" / f"{sid}.npy").resolve()),
                str((base / "esm2_layer_26" / "weighted_vecs" / f"{sid}.npy").resolve()),
                str((base / "esm2_layer_29" / "mean_vecs" / f"{sid}.npy").resolve()),
                str((base / "esm2_layer_29" / "weighted_vecs" / f"{sid}.npy").resolve()),
            }
            for sid in seq_ids
        }
        esmc_paths = {
            sid: {
                str((base / "esmc_layer_24" / "mean_vecs" / f"{sid}.npy").resolve()),
                str((base / "esmc_layer_24" / "weighted_vecs" / f"{sid}.npy").resolve()),
                str((base / "esmc_layer_32" / "mean_vecs" / f"{sid}.npy").resolve()),
                str((base / "esmc_layer_32" / "weighted_vecs" / f"{sid}.npy").resolve()),
            }
            for sid in seq_ids
        }
        prott5_paths = {
            sid: {
                str((base / "prot_t5_layer_19" / "mean_vecs" / f"{sid}.npy").resolve()),
                str((base / "prot_t5_layer_19" / "weighted_vecs" / f"{sid}.npy").resolve()),
                str((base / "prot_t5_last" / "mean_vecs" / f"{sid}.npy").resolve()),
                str((base / "prot_t5_last" / "weighted_vecs" / f"{sid}.npy").resolve()),
            }
            for sid in seq_ids
        }

        # kinform_t5_full covers: binding site prediction + T5 mean + T5 weighted.
        # It must run first so ESM2/ESMC weighted steps have binding sites available.
        bs_pred_path = media_path / "pseq2sites" / "binding_sites_all.tsv"
        seen_ids = _load_binding_site_ids(bs_pred_path)
        missing_bs = {sid for sid in seq_ids if sid not in seen_ids}
        plan_env = env or {}
        raw = str(plan_env.get("KINFORM_PARALLEL_EMBED_ENABLE", "1")).strip().lower()
        parallel_enabled = raw not in {"0", "false", "no", "off"}

        if parallel_enabled:
            if full_expected_paths is None:
                full_expected_paths = {}
            missing_all = missing_paths_by_seq_from_snapshot(full_expected_paths)
            missing_any_path = {
                sid
                for sid, paths in full_expected_paths.items()
                if sid in missing_all and paths
            }
            orchestrated_sids = sorted(missing_any_path | missing_bs)
            return [
                EmbeddingStepPlan(
                    step_key="kinform_t5_full",
                    missing_seq_ids=orchestrated_sids,
                    required_paths_by_seq={
                        sid: full_expected_paths.get(sid, set()) for sid in orchestrated_sids
                    },
                )
            ]

        missing_t5 = set(missing_paths_by_seq_from_snapshot(prott5_paths).keys())
        t5_full_sids = sorted(missing_bs | missing_t5)

        return [
            EmbeddingStepPlan(
                step_key="kinform_t5_full",
                missing_seq_ids=t5_full_sids,
                required_paths_by_seq={sid: prott5_paths[sid] for sid in t5_full_sids},
            ),
            _step_from_paths("kinform_esm2_layers", esm2_paths),
            _step_from_paths("kinform_esmc_layers", esmc_paths),
        ]

    return []


def missing_paths_by_seq_from_snapshot(expected: dict[str, set[str]]) -> dict[str, set[str]]:
    """Resolve missing artifacts without per-path filesystem probes.

    We use per-directory manifest reads with one directory snapshot fallback
    per cache bucket instead of calling Path.exists() for every expected path.
    """
    if not expected:
        return {}

    group_to_seq_ids: dict[tuple[Path, str], set[str]] = {}
    path_to_group_and_seq: dict[str, tuple[tuple[Path, str], str]] = {}

    for paths in expected.values():
        for path_str in paths:
            path = Path(path_str)
            group_key = (path.parent, path.suffix)
            seq_id = path.stem
            group_to_seq_ids.setdefault(group_key, set()).add(seq_id)
            path_to_group_and_seq[path_str] = (group_key, seq_id)

    group_to_ready_ids: dict[tuple[Path, str], set[str]] = {}
    for group_key, seq_ids in group_to_seq_ids.items():
        cache_dir, suffix = group_key
        _missing, ready = resolve_missing_ids(
            seq_ids,
            cache_dir=cache_dir,
            suffix=suffix,
        )
        group_to_ready_ids[group_key] = ready

    missing_by_seq: dict[str, set[str]] = {}
    for seq_id, paths in expected.items():
        seq_missing: set[str] = set()
        for path_str in paths:
            group_key, path_seq_id = path_to_group_and_seq[path_str]
            if path_seq_id not in group_to_ready_ids.get(group_key, set()):
                seq_missing.add(path_str)
        if seq_missing:
            missing_by_seq[seq_id] = seq_missing

    return missing_by_seq


def build_embedding_plan(
    *,
    method_key: str,
    target: str,
    sequences: list[str],
    env: dict,
) -> EmbeddingPlan:
    media_path, tools_path = resolve_media_and_tools(method_key, env)
    unique_sequences = normalise_sequences_for_method(method_key, sequences)
    if not unique_sequences:
        raise RuntimeError("No valid sequences for embedding planning.")

    seq_ids = resolve_seq_ids_via_cli(unique_sequences, tools_path, media_path)
    seq_id_to_seq = {sid: seq for sid, seq in zip(seq_ids, unique_sequences)}

    expected = expected_paths_by_seq(
        method_key=method_key,
        target=target,
        seq_ids=seq_ids,
        media_path=media_path,
        env=env,
    )
    if not expected:
        raise RuntimeError("No embedding cache profile for this method/target.")

    (
        missing_paths_by_seq,
        path_to_seqs,
        watch_dirs,
        total,
        cached_already,
        need_computation,
    ) = _partition_missing(expected)

    profile, gpu_supported, gpu_reason = _profile_for_method(method_key)
    step_plans = _step_plans_for_profile(
        profile=profile,
        method_key=method_key,
        target=target,
        seq_ids=seq_ids,
        media_path=media_path,
        env=env,
        full_expected_paths=expected,
    )

    return EmbeddingPlan(
        method_key=method_key,
        target=target,
        profile=profile,
        gpu_supported=gpu_supported,
        gpu_reason=gpu_reason,
        media_path=media_path,
        tools_path=tools_path,
        sequences=unique_sequences,
        seq_ids=seq_ids,
        seq_id_to_seq=seq_id_to_seq,
        expected_paths_by_seq=expected,
        missing_paths_by_seq=missing_paths_by_seq,
        path_to_seqs=path_to_seqs,
        watch_dirs=watch_dirs,
        total=total,
        cached_already=cached_already,
        need_computation=need_computation,
        step_plans=step_plans,
    )


def gpu_step_work(plan: EmbeddingPlan) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for step in plan.step_plans:
        if not step.gpu_enabled:
            continue
        if step.missing_seq_ids:
            out[step.step_key] = list(step.missing_seq_ids)
    return out
