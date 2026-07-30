#!/usr/bin/env python3
"""Notebook-friendly wrappers for the CatRange/CatPred/DLKcat/UniKP/EITLEM benchmark workflow."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import build_realkcat_kcat_benchmark_suite as build_suite_mod


def _resolve(pathlike) -> Path:
    return Path(pathlike).expanduser().absolute()


def _print_command(cmd: Sequence[object]) -> None:
    rendered = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"$ {rendered}")


def run_subprocess(cmd: Sequence[object], cwd=None, env=None) -> None:
    _print_command(cmd)
    subprocess.run(
        [str(part) for part in cmd],
        cwd=str(_resolve(cwd)) if cwd else None,
        env=env,
        check=True,
    )


def _env_with_pythonpath(extra_path) -> dict:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    extra = str(_resolve(extra_path))
    env["PYTHONPATH"] = extra if not current else f"{extra}:{current}"
    return env


def _probe_torch_runtime(python_executable) -> dict:
    probe_cmd = [
        str(_resolve(python_executable)),
        "-c",
        (
            "import json\n"
            "payload = {'import_ok': False, 'cuda_available': False, 'torch_version': None, 'cuda_version': None}\n"
            "try:\n"
            "    import torch\n"
            "    payload.update({\n"
            "        'import_ok': True,\n"
            "        'cuda_available': bool(torch.cuda.is_available()),\n"
            "        'torch_version': getattr(torch, '__version__', None),\n"
            "        'cuda_version': getattr(torch.version, 'cuda', None),\n"
            "    })\n"
            "except Exception as exc:\n"
            "    payload['error'] = str(exc)\n"
            "print(json.dumps(payload))\n"
        ),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {
            "import_ok": False,
            "cuda_available": False,
            "error": result.stderr.strip() or result.stdout.strip() or f"probe exited {result.returncode}",
        }
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {
            "import_ok": False,
            "cuda_available": False,
            "error": f"unparseable torch probe output: {result.stdout.strip()}",
        }


def resolve_torch_device(python_executable, requested: str = "auto", label: str = "model") -> str:
    requested = str(requested).strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        return requested
    if requested == "cpu":
        print(f"{label}: using requested device cpu")
        return "cpu"

    runtime = _probe_torch_runtime(python_executable)
    cuda_available = bool(runtime.get("cuda_available"))
    torch_version = runtime.get("torch_version") or "unknown"
    cuda_version = runtime.get("cuda_version") or "cpu-only"

    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
        print(
            f"{label}: resolved device {resolved} from {python_executable} "
            f"(torch={torch_version}, cuda={cuda_version})"
        )
        return resolved

    if cuda_available:
        print(
            f"{label}: using requested device cuda from {python_executable} "
            f"(torch={torch_version}, cuda={cuda_version})"
        )
        return "cuda"

    reason = runtime.get("error") or f"torch={torch_version}, cuda={cuda_version}"
    print(f"{label}: requested cuda but target env has no usable CUDA runtime ({reason}); falling back to cpu")
    return "cpu"


def resolve_torch_gpu_flag(python_executable, requested: bool, label: str = "model") -> bool:
    if not requested:
        print(f"{label}: GPU disabled by configuration")
        return False
    resolved = resolve_torch_device(python_executable, requested="cuda", label=label)
    return resolved == "cuda"


def _filter_specs(set_names: Optional[Iterable[str]] = None):
    if not set_names:
        return list(build_suite_mod._SET_SPECS)
    keep = {str(name) for name in set_names}
    return [spec for spec in build_suite_mod._SET_SPECS if str(spec["set_name"]) in keep]


def build_suite(
    realkcat_root,
    output_dir,
    parameter: str = "kcat",
    embedding: str = "esmc",
    set_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Build the benchmark suite from the exact CatRange fold workbooks."""
    realkcat_root = _resolve(realkcat_root)
    output_dir = build_suite_mod.make_output_dir(_resolve(output_dir))
    master_path = realkcat_root / build_suite_mod._MASTER_RAW_FILE
    if not master_path.exists():
        raise FileNotFoundError(f"CatRange master workbook not found: {master_path}")

    master_df = pd.read_excel(master_path, sheet_name="kcat_km_entries")
    master_df = master_df.reset_index().rename(columns={"index": "original_df_idx"})

    rows = []
    specs = _filter_specs(set_names)
    if not specs:
        raise ValueError("No benchmark set specifications matched the requested set_names.")

    for spec in specs:
        row = build_suite_mod._build_set(
            realkcat_root=realkcat_root,
            master_df=master_df,
            output_dir=output_dir,
            parameter=parameter,
            embedding=embedding,
            spec=spec,
        )
        rows.append(row)
        print(
            f"Built {row['set_name']}: n={row['n_rows']} rows, "
            f"{row['n_unique_sequences']} unique sequences"
        )

    manifest_df = pd.DataFrame(rows).sort_values(["comparison_group", "fold", "seqid_threshold", "set_name"])
    manifest_path = output_dir / "suite_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    build_suite_mod.write_json(
        output_dir / "suite_metadata.json",
        {
            "parameter": parameter,
            "embedding": embedding,
            "n_sets": int(len(manifest_df)),
            "manifest_csv": str(manifest_path.resolve()),
        },
    )
    print(f"Wrote suite manifest: {manifest_path}")
    return manifest_df


def run_realkcat_predictions(
    realkcat_root,
    suite_dir,
    python_executable,
    parameter: str = "kcat",
    embedding: str = "esmc",
    model_name: str = "CatRange",
):
    """Run CatRange predictions for every suite set."""
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "predict_realkcat_suite.py",
        "--realkcat-root",
        _resolve(realkcat_root),
        "--suite-dir",
        _resolve(suite_dir),
        "--parameter",
        parameter,
        "--embedding",
        embedding,
        "--model-name",
        model_name,
    ]
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted(_resolve(suite_dir).glob("sets/*/predictions/realkcat_standardized.csv"))


def build_catpred_protein_records(
    suite_dir,
    python_executable,
    esm_vendor,
    device: str = "auto",
    max_batch_size: int = 4,
    max_batch_tokens: int = 6000,
    cache_dir=None,
) -> pd.DataFrame:
    """Build CatPred `protein_records.json.gz` files for the suite."""
    device = resolve_torch_device(python_executable, requested=device, label="CatPred ESM2 embeddings")
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "build_catpred_protein_records.py",
        "--suite-dir",
        _resolve(suite_dir),
        "--device",
        device,
        "--max-batch-size",
        str(max_batch_size),
        "--max-batch-tokens",
        str(max_batch_tokens),
    ]
    if cache_dir:
        cmd.extend(["--cache-dir", _resolve(cache_dir)])
    run_subprocess(cmd, cwd=WORKSPACE_ROOT, env=_env_with_pythonpath(esm_vendor))
    return pd.read_csv(_resolve(suite_dir) / "suite_manifest.csv")


def run_catpred_predictions(
    suite_dir,
    checkpoint_dir,
    repo_root,
    python_executable,
    parameter: str = "kcat",
    use_gpu: bool = False,
    overwrite: bool = False,
    set_names: Optional[Iterable[str]] = None,
):
    """Run CatPred predictions for the suite."""
    use_gpu = resolve_torch_gpu_flag(python_executable, requested=use_gpu, label="CatPred inference")
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "run_catpred_suite.py",
        "--suite-dir",
        _resolve(suite_dir),
        "--checkpoint-dir",
        _resolve(checkpoint_dir),
        "--repo-root",
        _resolve(repo_root),
        "--parameter",
        parameter,
    ]
    if use_gpu:
        cmd.append("--use-gpu")
    if overwrite:
        cmd.append("--overwrite")
    for set_name in set_names or []:
        cmd.extend(["--set-name", str(set_name)])
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted(_resolve(suite_dir).glob("sets/*/predictions/catpred_standardized.csv"))


def run_catpred_retrained_predictions(
    suite_dir,
    realkcat_root,
    repo_root,
    python_executable,
    parameter: str = "kcat",
    embedding: str = "esmc",
    esm_vendor=None,
    device: str = "auto",
    gpu_index: int = 0,
    epochs: int = 30,
    ensemble_size: int = 1,
    batch_size: int = 16,
    seq_embed_dim: int = 36,
    seq_self_attn_nheads: int = 6,
    loss_function: str = "mve",
    max_batch_size: int = 4,
    max_batch_tokens: int = 6000,
    folds=None,
    overwrite: bool = False,
):
    """Retrain CatPred per CatRange fold and predict all suite sets."""
    device = resolve_torch_device(python_executable, requested=device, label="CatPred retraining")
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "run_catpred_retrain_suite.py",
        "--suite-dir",
        _resolve(suite_dir),
        "--realkcat-root",
        _resolve(realkcat_root),
        "--repo-root",
        _resolve(repo_root),
        "--python-executable",
        _resolve(python_executable),
        "--parameter",
        parameter,
        "--embedding",
        embedding,
        "--device",
        device,
        "--gpu-index",
        str(gpu_index),
        "--epochs",
        str(epochs),
        "--ensemble-size",
        str(ensemble_size),
        "--batch-size",
        str(batch_size),
        "--seq-embed-dim",
        str(seq_embed_dim),
        "--seq-self-attn-nheads",
        str(seq_self_attn_nheads),
        "--loss-function",
        str(loss_function),
        "--max-batch-size",
        str(max_batch_size),
        "--max-batch-tokens",
        str(max_batch_tokens),
    ]
    if esm_vendor:
        cmd.extend(["--esm-vendor", _resolve(esm_vendor)])
    for fold in folds or []:
        cmd.extend(["--fold", str(fold)])
    if overwrite:
        cmd.append("--overwrite")
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted(_resolve(suite_dir).glob("sets/*/predictions/catpred_standardized.csv"))


def run_dlkcat_predictions(
    suite_dir,
    dlkcat_root,
    model_python,
    controller_python,
):
    """Run DLKcat predictions for the suite."""
    cmd = [
        _resolve(controller_python),
        WORKSPACE_ROOT / "run_dlkcat_suite.py",
        "--suite-dir",
        _resolve(suite_dir),
        "--dlkcat-root",
        _resolve(dlkcat_root),
        "--python-executable",
        _resolve(model_python),
    ]
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted(_resolve(suite_dir).glob("sets/*/predictions/dlkcat_standardized.csv"))


def run_dlkcat_retrained_predictions(
    suite_dir,
    realkcat_root,
    dlkcat_root,
    python_executable,
    parameter: str = "kcat",
    embedding: str = "esmc",
    device: str = "auto",
    epochs: int = 50,
    min_epochs: int = 10,
    patience: int = 10,
    min_delta: float = 0.0,
    dim: int = 20,
    layer_gnn: int = 3,
    window: int = 11,
    layer_cnn: int = 3,
    layer_output: int = 3,
    learning_rate: float = 1e-3,
    lr_decay: float = 0.5,
    decay_interval: int = 10,
    weight_decay: float = 1e-6,
    folds=None,
    overwrite: bool = False,
):
    """Retrain DLKcat per CatRange fold and predict all suite sets."""
    device = resolve_torch_device(python_executable, requested=device, label="DLKcat retraining")
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "run_dlkcat_retrain_suite.py",
        "--suite-dir",
        _resolve(suite_dir),
        "--realkcat-root",
        _resolve(realkcat_root),
        "--dlkcat-root",
        _resolve(dlkcat_root),
        "--parameter",
        parameter,
        "--embedding",
        embedding,
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--min-epochs",
        str(min_epochs),
        "--patience",
        str(patience),
        "--min-delta",
        str(min_delta),
        "--dim",
        str(dim),
        "--layer-gnn",
        str(layer_gnn),
        "--window",
        str(window),
        "--layer-cnn",
        str(layer_cnn),
        "--layer-output",
        str(layer_output),
        "--learning-rate",
        str(learning_rate),
        "--lr-decay",
        str(lr_decay),
        "--decay-interval",
        str(decay_interval),
        "--weight-decay",
        str(weight_decay),
    ]
    for fold in folds or []:
        cmd.extend(["--fold", str(fold)])
    if overwrite:
        cmd.append("--overwrite")
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted(_resolve(suite_dir).glob("sets/*/predictions/dlkcat_standardized.csv"))


def run_unikp_predictions(
    suite_dir,
    realkcat_root,
    unikp_root,
    python_executable,
    parameter: str = "kcat",
    embedding: str = "esmc",
    device: str = "auto",
    n_estimators: int = 1000,
    n_jobs: int = -1,
    seq_batch_size: int = 2,
    smiles_batch_size: int = 256,
    exclude_val_from_train: bool = False,
    folds=None,
    overwrite: bool = False,
):
    """Train UniKP per fold and predict all suite sets for those folds."""
    device = resolve_torch_device(python_executable, requested=device, label="UniKP ProtT5")
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "run_unikp_suite.py",
        "--suite-dir",
        _resolve(suite_dir),
        "--realkcat-root",
        _resolve(realkcat_root),
        "--unikp-root",
        _resolve(unikp_root),
        "--parameter",
        parameter,
        "--embedding",
        embedding,
        "--device",
        device,
        "--n-estimators",
        str(n_estimators),
        "--n-jobs",
        str(n_jobs),
        "--seq-batch-size",
        str(seq_batch_size),
        "--smiles-batch-size",
        str(smiles_batch_size),
    ]
    for fold in folds or []:
        cmd.extend(["--fold", str(fold)])
    if exclude_val_from_train:
        cmd.append("--exclude-val-from-train")
    if overwrite:
        cmd.append("--overwrite")
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted(_resolve(suite_dir).glob("sets/*/predictions/unikp_standardized.csv"))


def run_eitlem_predictions(
    suite_dir,
    realkcat_root,
    eitlem_root,
    python_executable,
    parameter: str = "kcat",
    embedding: str = "esmc",
    device: str = "auto",
    epochs: int = 30,
    min_epochs: int = 5,
    patience: int = 5,
    min_delta: float = 1e-4,
    batch_size: int = 128,
    seq_batch_size: int = 1,
    num_workers: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    folds=None,
    overwrite: bool = False,
):
    """Train the EITLEM core predictor per fold and predict all suite sets."""
    device = resolve_torch_device(python_executable, requested=device, label="EITLEM")
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "run_eitlem_suite.py",
        "--suite-dir",
        _resolve(suite_dir),
        "--realkcat-root",
        _resolve(realkcat_root),
        "--eitlem-root",
        _resolve(eitlem_root),
        "--parameter",
        parameter,
        "--embedding",
        embedding,
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--min-epochs",
        str(min_epochs),
        "--patience",
        str(patience),
        "--min-delta",
        str(min_delta),
        "--batch-size",
        str(batch_size),
        "--seq-batch-size",
        str(seq_batch_size),
        "--num-workers",
        str(num_workers),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
    ]
    for fold in folds or []:
        cmd.extend(["--fold", str(fold)])
    if overwrite:
        cmd.append("--overwrite")
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted(_resolve(suite_dir).glob("sets/*/predictions/eitlem_standardized.csv"))


def evaluate_suite(suite_dir, python_executable):
    """Evaluate the finished benchmark suite and load its summary tables."""
    suite_dir = _resolve(suite_dir)
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "evaluate_benchmark_suite.py",
        "--suite-dir",
        suite_dir,
    ]
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return load_suite_results(suite_dir)


def plot_suite_figures(suite_dir, python_executable, dpi: int = 600):
    """Create single-panel publication figures for the suite."""
    suite_dir = _resolve(suite_dir)
    cmd = [
        _resolve(python_executable),
        WORKSPACE_ROOT / "plot_publication_single_figures.py",
        "--suite-dir",
        suite_dir,
        "--dpi",
        str(dpi),
    ]
    run_subprocess(cmd, cwd=WORKSPACE_ROOT)
    return sorted((suite_dir / "suite_results" / "figures").glob("*"))


def load_suite_results(suite_dir):
    """Load the suite summary tables into a small dictionary."""
    suite_dir = _resolve(suite_dir)
    results_dir = suite_dir / "suite_results"
    return {
        "results_dir": results_dir,
        "all_set_summary": pd.read_csv(results_dir / "all_set_summary.csv"),
        "fold_test_summary": pd.read_csv(results_dir / "fold_test_summary.csv"),
        "fold_test_mean_std": pd.read_csv(results_dir / "fold_test_mean_std.csv"),
        "fold5_test_summary": pd.read_csv(results_dir / "fold5_test_summary.csv"),
        "fold5_seqid_summary": pd.read_csv(results_dir / "fold5_seqid_summary.csv"),
    }


def load_catpred_coverage(suite_dir) -> pd.DataFrame:
    """Collect CatPred coverage diagnostics across suite sets."""
    suite_dir = _resolve(suite_dir)
    rows = []
    for path in sorted(suite_dir.glob("sets/*/predictions/catpred_coverage.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "set_name": path.parent.parent.name,
                "total_rows": payload["total_rows"],
                "valid_rows": payload["valid_rows"],
                "skipped_rows": payload["skipped_rows"],
                "skipped_fraction": payload["skipped_fraction"],
            }
        )
    return pd.DataFrame(rows).sort_values("set_name") if rows else pd.DataFrame()


def list_prediction_outputs(suite_dir, model_slug: str):
    """List standardized prediction outputs for a given model slug."""
    return sorted(_resolve(suite_dir).glob(f"sets/*/predictions/{model_slug}_standardized.csv"))
