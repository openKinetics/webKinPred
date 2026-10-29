import numpy as np
import pytest
from api.execution.validation import gpu_numerical_waiver, parity_accepted
from tools.executor.policy import planned_sequence_ids
from tools.executor.validate_artifacts import metadata


@pytest.mark.parametrize("use_host", [True, False])
def test_canary_healthy_hour_restarts_after_recovery_without_losing_completed_jobs(use_host):
    from datetime import timedelta
    from types import SimpleNamespace

    from api.management.commands.execution_readiness import canary_timestamps
    from django.utils import timezone

    now = timezone.now()
    start = now - timedelta(hours=2)
    recovery = now - timedelta(minutes=15)
    evidence = {"canary_started_at": start.isoformat()}
    hosts = []
    if use_host:
        hosts = [SimpleNamespace(telemetry={"control_recovered_at": recovery.isoformat()})]
    else:
        evidence["canary_healthy_since"] = recovery.isoformat()
    completed_after, observed_after = canary_timestamps(evidence, hosts)
    assert completed_after == start
    assert now - observed_after < timedelta(hours=1)


def test_profiling_disk_sample_survives_atomic_output_rename(tmp_path):
    from types import SimpleNamespace

    from tools.executor.profile import scratch_bytes

    def vanished():
        raise FileNotFoundError("Concurrent atomic publication")

    paths = [
        SimpleNamespace(is_file=lambda: True, stat=vanished),
        SimpleNamespace(is_file=lambda: True, stat=lambda: SimpleNamespace(st_size=42)),
    ]
    assert scratch_bytes(SimpleNamespace(rglob=lambda _: iter(paths))) == 42


def evidence(**changes):
    return {
        "gpu_numerical_waiver": {
            "approved_by_user": True,
            "scope": "gpu_numerical_parity",
            "request_reference": "2026-10-28 deployment instruction",
            "shape_dtype_finite_passed": True,
            **changes,
        }
    }


def test_explicit_gpu_waiver_does_not_waive_cpu_compatibility():
    assert parity_accepted(evidence(), gpu=True)
    assert not parity_accepted(evidence(), gpu=False)
    assert parity_accepted({"parity_passed": True}, gpu=False)


@pytest.mark.parametrize(
    "message",
    [
        "DefaultCPUAllocator: can't allocate memory",
        "std::bad_alloc",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_native_allocator_errors_are_recoverable_oom(message):
    from api.execution.errors import classify

    assert classify(RuntimeError(message)) == "oom"


def test_numerical_waiver_still_requires_structural_checks_and_attribution():
    assert not gpu_numerical_waiver(evidence(shape_dtype_finite_passed=False))
    assert not gpu_numerical_waiver(evidence(request_reference=""))
    assert not gpu_numerical_waiver(evidence(approved_by_user=False))
    assert not gpu_numerical_waiver(evidence(scope="all_validation"))


def test_generated_arrays_must_be_finite_and_report_their_representation(tmp_path):
    path = tmp_path / "embedding.npy"
    request = {"payload": {"step": "prot_t5_mean"}}
    np.save(path, np.ones(1024, dtype=np.float32))
    assert metadata(path, request) == {"shape": [1024], "dtype": "float32", "finite": True}
    np.save(path, np.asarray([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="Non-finite"):
        metadata(path, request)
    np.save(path, np.asarray([1], dtype=np.int32))
    with pytest.raises(ValueError, match="non-floating"):
        metadata(path, request)


def test_binding_site_publication_rejects_partial_sequence_rows(tmp_path):
    path = tmp_path / "binding.tsv"
    request = {"payload": {"step": "kinform_pseq2sites", "seq_id_to_seq": {"sid": "MAAA"}}}
    path.write_text("PDB\tPred_BS_Scores\nsid\t0.1,0.2\n")
    with pytest.raises(ValueError, match="Invalid binding-site"):
        metadata(path, request)
    path.write_text("PDB\tPred_BS_Scores\nsid\t0.1,0.2,0.3,0.4\n")
    assert metadata(path, request)["rows"] == 1


def test_attempt_sequence_ids_preserve_collisions_and_reject_unplanned_sequences(
    tmp_path, monkeypatch
):
    path = tmp_path / "seqmap.json"
    path.write_text('{"MAAA": "historic_123_2", "MBBB": "historic_123"}')
    monkeypatch.setenv("SEQMAP_MANIFEST_PATH", str(path))
    assert planned_sequence_ids(["MBBB", "MAAA", "MBBB"]) == [
        "historic_123",
        "historic_123_2",
        "historic_123",
    ]
    with pytest.raises(KeyError):
        planned_sequence_ids(["MCCC"])


def test_binding_merge_preserves_rows_and_rejects_duplicate_partitions(tmp_path):
    from tools.executor.kinform import merge_binding

    (tmp_path / "one.tsv").write_text("PDB\tPred_BS_Scores\na\t0.1,0.2\n")
    (tmp_path / "two.tsv").write_text("PDB\tPred_BS_Scores\nb\t0.3\n")
    request = {
        "payload": {
            "parts": ["one.tsv", "two.tsv"],
            "seq_id_to_seq": {"a": "MA", "b": "M"},
            "outputs": [{"path": "merged.tsv"}],
            "step": "kinform_binding_merge",
        }
    }
    merge_binding(request, tmp_path)
    assert metadata(tmp_path / "merged.tsv", request)["rows"] == 2
    request["payload"]["parts"].append("one.tsv")
    with pytest.raises(ValueError, match="Duplicate"):
        merge_binding(request, tmp_path)


def test_tensor_checkpoints_preserve_complete_archives_but_reject_interrupted_writes(tmp_path):
    import torch
    from tools.executor.checkpoints import complete_tensor_archive, partial_manifest

    path = tmp_path / "media/tensor.pt"
    path.parent.mkdir()
    torch.save(torch.ones((128, 1316)), path)
    assert complete_tensor_archive(path)
    request = {
        "kind": "embedding",
        "method": "CatPred",
        "payload": {"outputs": [{"id": "tensor", "identity": {}, "path": "tensor.pt"}]},
    }
    assert len(partial_manifest(request, tmp_path)["artifacts"]) == 1
    path.write_bytes(path.read_bytes()[:-32])
    assert not complete_tensor_archive(path)
    assert not partial_manifest(request, tmp_path)["artifacts"]


def test_profile_envelope_includes_shared_pages_and_unsampled_allocator_peaks():
    from tools.executor.profiling_suite import measurement_peaks

    report = {
        "peaks": {"ram": 100, "vram": 200},
        "samples": [{"process_tree": {"rss": 500}}],
        "framework_metrics": [{"reserved_peak": 300}, {"reserved_peak": 100}],
    }
    assert measurement_peaks(report) == {"ram": 500, "vram": 400}


def test_turnup_service_refuses_missing_cuda_before_importing_the_encoder(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tools.gpu_embed_service import run_step

    monkeypatch.delenv("WEBKINPRED_PROFILING", raising=False)
    monkeypatch.setenv("GPU_REPO_ROOT", str(tmp_path))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )

    def run(command, env):
        exec(compile(command[2], "isolated-turnup-worker", "exec"), {})

    monkeypatch.setattr(run_step, "_run", run)
    with pytest.raises(RuntimeError, match="GPU_UNAVAILABLE"):
        run_step._run_turnup(
            {
                "KINFORM_MEDIA_PATH": str(tmp_path),
                "KINFORM_TOOLS_PATH": str(tmp_path),
                "GPU_REPO_ROOT": str(tmp_path),
            },
            tmp_path / "sequences.json",
        )


def test_turnup_retry_profile_controls_the_actual_encoder_batch(tmp_path, monkeypatch):
    import os

    from tools.executor import runner

    monkeypatch.setenv("TURNUP_EMBED_BATCH_SIZE", "8")
    monkeypatch.setattr(runner, "atomic_json", lambda *args: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: None)
    runner.embedding(
        {
            "id": "probe",
            "settings": {"batch_size": 2},
            "payload": {"step": "turnup_esm1b", "seq_id_to_seq": {"sid": "MAAA"}},
        },
        tmp_path,
    )
    assert os.environ["TURNUP_EMBED_BATCH_SIZE"] == "2"


@pytest.mark.parametrize(
    "fault", [None, "uncommitted", "no_gpu_residency", "budget", "unapproved", "delayed", "stale"]
)
def test_parallel_ramp_requires_successful_actual_residency_and_bounded_reservations(
    tmp_path, fault
):
    import json
    from datetime import timedelta
    from types import SimpleNamespace

    from api.management.commands.execution_readiness import parallel_ramp_errors
    from django.utils import timezone

    now = timezone.now()
    path = tmp_path / "samples.jsonl"
    attempts = {}
    hosts = []
    for name in ["production", "gpu"]:
        rows = []
        for index in range(4 if name == "gpu" else 2):
            ident = name + str(index)
            gpu = index >= 2
            resources = {"ram": 10, "cpu": 1, "disk": 1, "vram": 10 if gpu else 0}
            attempts[ident] = SimpleNamespace(
                state="completed",
                host_id=name,
                resources=resources,
                started_at=now - timedelta(seconds=5),
                finished_at=now + timedelta(seconds=10),
            )
            rows.append(
                {
                    "id": ident,
                    "state": "running",
                    "resources": resources,
                    "observed": {"ram": 5, "vram": 5 if gpu else 0},
                }
            )
        hosts.append({"host": name, "budgets": {"cpu": 4, "ram": 40, "disk": 4}, "attempts": rows})
    evidence = {
        "rollout_policy": "operator_authorized_parallel_ramp",
        "rollout_authorization": "user request",
        "parallel_ramp_started_at": (now - timedelta(seconds=10)).isoformat(),
        "parallel_samples": str(path),
    }
    if fault == "uncommitted":
        attempts["gpu2"].state = "running"
    elif fault == "no_gpu_residency":
        hosts[1]["attempts"][2]["observed"]["vram"] = 0
    elif fault == "budget":
        hosts[1]["budgets"]["ram"] = 30
    elif fault == "unapproved":
        evidence["rollout_authorization"] = ""
    if fault == "delayed":
        for attempt in attempts.values():
            attempt.finished_at = now + timedelta(seconds=3)
    lines = []
    for i in range(3):
        observed = now + timedelta(seconds=i)
        if fault in {"delayed", "stale"}:
            for host in hosts:
                host["heartbeat"] = (
                    observed - timedelta(seconds=20 if fault == "stale" else 0)
                ).isoformat()
        collected = observed + timedelta(seconds=5 if fault == "delayed" else 0)
        lines.append(json.dumps({"at": collected.isoformat(), "hosts": hosts}) + "\n")
    path.write_text("".join(lines))
    errors = parallel_ramp_errors(evidence, attempts)
    assert bool(errors) == (fault not in {None, "delayed"})
