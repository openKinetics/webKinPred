from types import SimpleNamespace

import pytest
from api.execution import control
from api.execution.artifacts import sha256_file
from api.execution.models import ExecutorHost, QuotaDay, QuotaEntry
from api.execution.planning import prepare
from api.execution.quota import refund
from api.models import Job
from tools.executor.molecular import legacy_transformer_vectors


def specification(method="DLKcat"):
    return {
        "targets": ["kcat"],
        "methods": {"kcat": method},
        "model_versions": {method: {"weights": "a" * 64}},
        "canonicalize_substrates": True,
        "handle_long_sequences": "truncate",
        "include_similarity_columns": True,
        "experimental_results": {},
    }


@pytest.mark.django_db(transaction=True)
def test_preparation_is_idempotent_and_preserves_rows_and_dependencies(tmp_path):
    ExecutorHost.objects.create(id="production")
    job = Job.objects.create(
        public_id="prepare1", status="Pending", prediction_type="kcat", requested_rows=3
    )
    directory = tmp_path / "media/jobs/prepare1"
    directory.mkdir(parents=True)
    source = directory / "input.csv"
    source.write_text("Protein Sequence,Substrate\nMAAA,CC\nMCCC,O\nMDDD,N\n")
    execution = control.ingest(job.public_id, specification(), sha256_file(source))
    prepare(execution, tmp_path / "media", tmp_path / "blobs")
    first = list(execution.stages.values_list("id", "key"))
    prepare(execution, tmp_path / "media", tmp_path / "blobs")
    assert set(first) == set(execution.stages.values_list("id", "key"))
    assert set(execution.stages.values_list("key", flat=True)) == {
        "prepare",
        "predict:kcat:0",
        "assemble",
        "similarity",
        "publish",
    }
    # Planning scratch is reclaimable only after published chunks and graph commit.
    assert not (tmp_path / "planning" / str(execution.pk)).exists()
    assert source.exists() and sha256_file(source) == execution.input_checksum
    prediction = execution.stages.get(key="predict:kcat:0")
    input_artifact = prediction.artifacts.get(path="jobs/prepare1/input.csv").artifact
    assert (
        sha256_file(tmp_path / "blobs" / input_artifact.checksum[:2] / input_artifact.checksum)
        == input_artifact.checksum
    )
    assert prediction.payload["offset"] == 0
    assert prediction.payload["rows"] == 3
    assert prediction.artifacts.filter(path="jobs/prepare1/input.csv").exists()
    assert execution.stages.get(key="assemble").dependencies.filter(pk=prediction.pk).exists()


@pytest.mark.django_db(transaction=True)
def test_molecular_graph_preserves_context_and_has_no_hidden_cpu_embedding(tmp_path):
    ExecutorHost.objects.create(id="production")
    job = Job.objects.create(
        public_id="prepare2", status="Pending", prediction_type="kcat", requested_rows=2
    )
    directory = tmp_path / "media/jobs/prepare2"
    directory.mkdir(parents=True)
    source = directory / "input.csv"
    source.write_text("Protein Sequence,Substrate\nMAAA,CC\nMAAA,CC\n")
    execution = control.ingest(job.public_id, specification("UniKP"), sha256_file(source))
    prepare(execution, tmp_path / "media", tmp_path / "blobs")
    prediction = execution.stages.get(key="predict:kcat:0")
    rows = prediction.payload["molecular"]["UniKP"]
    assert [row["legacy_position"] for row in rows] == [0, 1]
    assert rows[0]["path"] != rows[1]["path"]
    assert not execution.stages.filter(state="held").exists()
    assert execution.stages.filter(kind="embedding", method="molecular-UniKP").count() == 1


@pytest.mark.django_db(transaction=True)
def test_kinform_binding_partitions_bound_gpu_work_and_merge_before_prediction():
    from api.execution.planning import kinform_graph

    job = Job.objects.create(public_id="kf-parts", status="Pending", prediction_type="kcat")
    execution = control.ingest(job.public_id, specification("KinForm-H"), "a" * 64)
    mapping = {"seq" + str(i): "M" + "A" * (i + 1) for i in range(35)}
    outputs = kinform_graph(execution, "KinForm-H", "kcat", mapping)
    binding = execution.stages.filter(payload__step="kinform_pseq2sites")
    assert sorted(s.payload["shape"]["sequences"] for s in binding) == [3, 16, 16]
    merge = execution.stages.get(payload__step="kinform_binding_merge")
    assert merge.kind == "predict" and merge.dependencies.count() == 3
    assert len(set(merge.payload["parts"])) == 3
    pools = execution.stages.filter(payload__step="kinform_pool")
    assert pools.count() == 3
    assert all(stage.dependencies.filter(pk=merge.pk).exists() for stage in pools)
    assert len(outputs) == 35 * 12 + 1
    before = execution.stages.count()
    kinform_graph(execution, "KinForm-H", "kcat", mapping)
    assert execution.stages.count() == before


@pytest.mark.django_db(transaction=True)
def test_refund_uses_original_quota_day_and_is_idempotent():
    from datetime import date

    job = Job.objects.create(public_id="quota1", status="Pending", prediction_type="kcat")
    old = QuotaDay.objects.create(subject="test", day=date(2025, 1, 1), used=20)
    today = QuotaDay.objects.create(subject="test", day=date(2026, 10, 28), used=30)
    QuotaEntry.objects.create(
        key="legacy-reservation:1", account=old, job=job, amount=0, reserved_amount=10
    )
    assert refund(job, 6) == 6
    assert refund(job, 6) == 0
    old.refresh_from_db()
    today.refresh_from_db()
    assert old.used == 14
    assert today.used == 30


def test_legacy_positions_survive_microbatch_splits():
    import torch

    torch.manual_seed(1)
    model = SimpleNamespace(
        embed=torch.nn.Embedding(8, 4),
        pe=SimpleNamespace(pe=torch.arange(100 * 4, dtype=torch.float32).reshape(1, 100, 4)),
        trfm=SimpleNamespace(
            encoder=SimpleNamespace(layers=[lambda x, _: x * 2, lambda x, _: x + 1], norm=None)
        ),
    )
    vocab = SimpleNamespace(stoi={"C": 4})
    rows = [{"input": "C", "legacy_position": n} for n in [0, 1, 99, 0, 27, 0]]
    whole = legacy_transformer_vectors(model, vocab, lambda x: x, rows, torch, device="cpu")
    import numpy as np

    split = np.concatenate(
        [
            legacy_transformer_vectors(model, vocab, lambda x: x, rows[:2], torch, device="cpu"),
            legacy_transformer_vectors(model, vocab, lambda x: x, rows[2:], torch, device="cpu"),
        ]
    )
    np.testing.assert_array_equal(whole, split)
    assert not np.array_equal(whole[0], whole[1])


def test_cpu_cache_miss_and_missing_cuda_never_fall_back(monkeypatch):
    from tools.executor.policy import require_cached, require_cuda

    monkeypatch.setenv("WEBKINPRED_EXECUTION_POLICY", "gpu-only")
    with pytest.raises(RuntimeError, match="MISSING_EMBEDDING"):
        require_cached(False, "test")
    with pytest.raises(RuntimeError, match="GPU_UNAVAILABLE"):
        require_cuda(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


def test_bounded_cpu_calls_keep_positions_errors_details_and_progress(monkeypatch):
    from api import tasks
    from api.execution.context import StageContext, current, stage_context

    selected, calls = [], []
    context = StageContext(
        "units",
        unit_batch_size=2,
        unit_context={str(i): {"position": i} for i in range(5)},
        select_units=lambda indices: selected.append(list(indices)),
    )
    descriptor = SimpleNamespace(key="probe", col_to_kwarg={"Substrates": "substrates"})

    def engine(desc, sequences, public_id, target, **kwargs):
        original = [current.get().unit_context[str(i)]["position"] for i in range(len(sequences))]
        calls.append((sequences, kwargs["substrates"], original))
        return original, ({0: "invalid"} if original[0] == 2 else {}), [str(i) for i in original]

    monkeypatch.setattr(tasks, "_run_method_engine", engine)
    with stage_context(context):
        values, errors, extra = tasks._invoke_method_prediction(
            descriptor,
            ["A", "C", "D", "E", "F"],
            "units",
            "kcat",
            substrates=["C", "CC", "CCC", "CCCC", "CCCCC"],
        )
    assert values == [0, 1, 2, 3, 4] and errors == {2: "invalid"}
    assert extra == ["0", "1", "2", "3", "4"]
    assert selected == [[0, 1], [2, 3], [4]]
    assert calls[1] == (["D", "E"], ["CCC", "CCCC"], [2, 3])
    assert context.progress == 5 and context.progress_base == 0
    assert context.unit_context["4"] == {"position": 4}


def test_invalid_rows_do_not_reserve_gpu_embedding_work():
    from api.execution.planning import embedding_sequences

    descriptor = SimpleNamespace(col_to_kwarg={"Substrate": "substrates"})
    batch = SimpleNamespace(
        sequences=["MAAA", "MAAX", "MCCC"], call_kwargs={"substrates": ["CC", "O", "not-smiles"]}
    )
    assert embedding_sequences(descriptor, batch) == ["MAAA"]


def test_embedding_reuse_preserves_target_specific_and_encoder_identities():
    from api.execution.planning import artifact_identity

    path = "sequence_info/prot_t5_last/mean_vecs/sequence.npy"
    spec = {
        "model_versions": {
            "UniKP": {"weights": "a"},
            "CataPro": {"weights": "b"},
            "CatPred": {"weights": "c"},
        }
    }
    assert artifact_identity(spec, "UniKP", "kcat", path) == artifact_identity(
        spec, "UniKP", "Km", path
    )
    assert artifact_identity(spec, "CatPred", "kcat", path) != artifact_identity(
        spec, "CatPred", "Km", path
    )
    assert artifact_identity(spec, "UniKP", "kcat", path) != artifact_identity(
        spec, "CataPro", "kcat", path
    )
    for method in ["UniKP", "CataPro"]:
        spec["model_versions"][method]["representations"] = {
            "sequence_info/prot_t5_last/mean_vecs": {"encoder": "identical", "precision": "fp16"}
        }
    assert artifact_identity(spec, "UniKP", "kcat", path) == artifact_identity(
        spec, "CataPro", "Km", path
    )


@pytest.mark.django_db(transaction=True)
def test_lost_embedding_is_regenerated_for_consumer_without_reopening_completed_job():
    from api.execution.models import Artifact, ArtifactLocation, WorkStage
    from api.execution.planning import attach
    from api.execution.recovery import missing_dependencies

    host = ExecutorHost.objects.create(id="gpu")
    old_job = Job.objects.create(public_id="old-done", status="Completed", prediction_type="kcat")
    old = control.ingest(old_job.public_id, specification("UniKP"), "a" * 64)
    old_stage = WorkStage.objects.create(
        execution=old, key="embedding", kind="embedding", method="prot_t5_mean", state="completed"
    )
    identity = {"representation": "pinned-t5", "sequence": "MAAA"}
    artifact = Artifact.objects.create(
        id=control.digest(identity),
        identity=identity,
        checksum="b" * 64,
        size=12,
        producer=old_stage,
    )
    old_stage.payload = {
        "outputs": [{"id": artifact.pk, "identity": identity, "path": "t5/a.npy"}],
        "output_artifacts": [artifact.pk],
        "step": "prot_t5_mean",
    }
    old_stage.save()
    ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
    job = Job.objects.create(public_id="new-user", status="Processing", prediction_type="kcat")
    execution = control.ingest(job.public_id, specification("UniKP"), "a" * 64)
    consumer = WorkStage.objects.create(
        execution=execution, key="head", kind="predict", method="UniKP", state="recovering"
    )
    attach(consumer, artifact, "t5/a.npy")
    missing_dependencies(consumer.pk, host.pk)
    consumer.refresh_from_db()
    artifact.refresh_from_db()
    old_job.refresh_from_db()
    restored = artifact.producer
    assert old_job.status == "Completed"
    assert old_stage.state == "completed"
    assert restored.execution_id == execution.pk
    assert restored.kind == "embedding" and restored.state == "ready"
    assert consumer.dependencies.filter(pk=restored.pk).exists()
    assert consumer.state == "ready" and consumer.waiting_reason == "waiting_embeddings"
    assert not artifact.locations.filter(ready=True).exists()
    assert artifact.checksum == "" and artifact.size == 0
    assert artifact.metadata["previous_checksums"] == ["b" * 64]
    missing_dependencies(consumer.pk, host.pk)
    assert execution.stages.filter(key="restore:" + artifact.pk).count() == 1


@pytest.mark.django_db(transaction=True)
def test_lost_original_input_is_held_without_recomputation():
    from api.execution.models import Artifact, WorkStage
    from api.execution.planning import attach
    from api.execution.recovery import missing_dependencies

    job = Job.objects.create(public_id="lost-input", status="Processing", prediction_type="kcat")
    execution = control.ingest(job.public_id, specification(), "a" * 64)
    stage = WorkStage.objects.create(
        execution=execution, key="head", kind="predict", state="recovering"
    )
    artifact = Artifact.objects.create(
        id="a" * 64, identity={"input": "original"}, checksum="b" * 64, size=12, regenerable=False
    )
    attach(stage, artifact, "jobs/lost-input/input.csv")
    missing_dependencies(stage.pk, None)
    stage.refresh_from_db()
    job.refresh_from_db()
    assert stage.state == "held" and stage.waiting_reason == "artifact_requires_attention"
    assert job.status == "Processing"


@pytest.mark.parametrize("method", ["OmniESI", "OmniESI-O2DENet"])
@pytest.mark.parametrize("unit_batch_size", [1, 128])
def test_all_invalid_native_batches_preserve_rows_without_loading_a_model(
    monkeypatch, method, unit_batch_size
):
    import pandas as pd
    from api.execution.context import StageContext, execution_job, stage_context
    from api.methods.registry import get
    from api.prediction_engines import generic_subprocess
    from api.tasks import _execute_target_batch

    def forbidden(*args, **kwargs):
        raise AssertionError("An all-invalid batch must not resolve or start a model")

    monkeypatch.setattr(generic_subprocess, "_resolve_subprocess_paths", forbidden)
    monkeypatch.setattr(generic_subprocess, "run_prediction_subprocess", forbidden)
    frame = pd.DataFrame(
        {"Protein Sequence": ["MAAA", "MCCC"], "Substrate": ["invalid-chemistry", "also-invalid"]}
    )
    with stage_context(StageContext("invalid-native", unit_batch_size=unit_batch_size)):
        result = _execute_target_batch(
            job=execution_job("invalid-native"),
            desc=get(method),
            df=frame,
            target="kcat",
            sequences=frame["Protein Sequence"].tolist(),
            processed_by_reaction=frame["Protein Sequence"].tolist(),
            valid_reaction_indices=[0, 1],
            experimental_results=[],
            canonicalize_substrates=True,
            disable_gpu_precompute=False,
            extra_call_kwargs={},
        )
    assert result["preds"] == ["", ""]
    assert result["extra"] == ["", ""]
    assert set(result["failed_reactions"]) == {0, 1}
    assert all("Invalid substrate" in value for value in result["sources"])
