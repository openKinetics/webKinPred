from copy import deepcopy

import pytest
from api.execution import control
from api.execution.models import ArtifactLocation, ExecutorHost, WorkStage
from api.execution.planning import artifact_identity, attach, declare, embedding_stage
from api.execution.representation_aliases import alternate_identity
from api.models import Job

pytestmark = pytest.mark.django_db(transaction=True)


def execution(method, name, inventory="same"):
    versions = {
        m: {
            "method": m,
            "representation_policy": "20261028-v2",
            "gpu_inventory": inventory,
            "production_inventory": "same",
            "protein_precision": "legacy",
            "molecular_precision": "fp32",
        }
        for m in ["KinForm-H", "KinForm-L", "OmniESI", "OmniESI-O2DENet"]
    }
    Job.objects.create(public_id=name, prediction_type="kcat", status="Pending")
    return control.ingest(
        name,
        {"targets": ["kcat"], "methods": {"kcat": method}, "model_versions": versions},
        "a" * 64,
        paused=False,
    )


@pytest.mark.parametrize(
    "first,second,path,step",
    [
        (
            "KinForm-H",
            "KinForm-L",
            "sequence_info/prot_t5_last/residue_vecs/seq.npy",
            "kinform_t5_residues",
        ),
        ("OmniESI", "OmniESI-O2DENet", "sequence_info/omniesi_esm2/seq.npy", "omniesi_esm2"),
    ],
)
def test_identical_encoder_work_has_one_producer_and_separate_jobs(first, second, path, step):
    left, right = execution(first, "aliasleft"), execution(second, "aliasright")
    before = deepcopy(right.specification)
    a = embedding_stage(left, first, "kcat", {"seq": "MAAA"}, {"seq": {path}}, {"step": step})[0][0]
    b = embedding_stage(right, second, "kcat", {"seq": "MAAA"}, {"seq": {path}}, {"step": step})[0][
        0
    ]
    assert a.pk == b.pk and a.producer_id == b.producer_id
    assert left.stages.filter(kind="embedding").count() == 1
    assert not right.stages.filter(kind="embedding").exists()
    head = WorkStage.objects.create(execution=right, key="prediction", kind="predict")
    attach(head, b, path)
    assert head.dependencies.filter(pk=a.producer_id).exists()
    right.refresh_from_db()
    assert right.specification == before and right.job_id != left.job_id


def test_aliases_never_mix_changed_models_contexts_or_existing_manifests():
    a, b = execution("KinForm-H", "first"), execution("KinForm-L", "second")
    path = "sequence_info/molecular/KinForm/molecule.npy"
    identity = artifact_identity(
        a.specification, "KinForm-H", "kcat", path, context={"input": "CC", "legacy_position": 1}
    )
    ready = declare(identity)
    ready.checksum = "b" * 64
    ready.save()
    host = ExecutorHost.objects.create(id="gpu")
    ArtifactLocation.objects.create(artifact=ready, host=host, ready=True)
    equivalent = alternate_identity(identity)
    assert declare(equivalent).pk == ready.pk
    changed = deepcopy(equivalent)
    changed["context"]["legacy_position"] = 99
    assert declare(changed).pk != ready.pk
    changed = deepcopy(equivalent)
    changed["models"]["gpu_inventory"] = "new"
    assert declare(changed).pk != ready.pk
    changed = deepcopy(equivalent)
    changed["models"]["molecular_precision"] = "fp16"
    assert declare(changed).pk != ready.pk
    # An already declared identity remains authoritative even if another ready
    # representation is subsequently discovered.
    existing = declare(changed)
    assert declare(changed).pk == existing.pk
    assert b.specification["methods"]["kcat"] == "KinForm-L"


def test_completed_missing_or_paused_encoder_does_not_block_an_unrelated_new_job():
    left, right = execution("KinForm-H", "older"), execution("KinForm-L", "newer")
    path = "sequence_info/esmc_layer_32/mean_vecs/seq.npy"
    a = embedding_stage(
        left, "KinForm-H", "kcat", {"seq": "MAAA"}, {"seq": {path}}, {"step": "kinform_pool"}
    )[0][0]
    left.paused = True
    left.save()
    identity = artifact_identity(right.specification, "KinForm-L", "kcat", path)
    assert declare(identity).pk != a.pk


def test_unknown_representations_and_catpred_targets_never_alias():
    e = execution("KinForm-H", "bounds")
    original = artifact_identity(
        e.specification, "KinForm-H", "kcat", "sequence_info/catrange_esmc/seq.npy"
    )
    assert alternate_identity(original) is None
    original["path"] = "sequence_info/esmc_layer_32/mean_vecs/seq.npy"
    original["target"] = "Km"
    assert alternate_identity(original) is None
    original["target"] = None
    original["models"]["representation_policy"] = "future-version"
    assert alternate_identity(original) is None


def test_kinform_sharing_creates_no_orphan_residue_or_pooling_work():
    from api.execution.planning import kinform_graph

    left, right = execution("KinForm-H", "graphleft"), execution("KinForm-L", "graphright")
    mapping = {"s1": "MAAA", "s2": "MCCC"}
    first = kinform_graph(left, "KinForm-H", "kcat", mapping)
    original = list(left.stages.order_by("pk").values("id", "kind", "method", "payload"))
    second = kinform_graph(right, "KinForm-L", "kcat", mapping)
    assert {a.pk for a, _, _ in first} == {a.pk for a, _, _ in second}
    assert not right.stages.exclude(kind="prepare").exists()
    assert list(left.stages.order_by("pk").values("id", "kind", "method", "payload")) == original


def test_partial_cached_vectors_only_plan_the_missing_encoder_family():
    from api.execution.planning import kinform_graph
    from api.execution.representation_aliases import KINFORM_ROOTS

    e = execution("KinForm-H", "partial")
    host = ExecutorHost.objects.create(id="gpu")
    missing = "sequence_info/esmc_layer_32/mean_vecs/seq.npy"
    for root in KINFORM_ROOTS:
        for kind in ["mean", "weighted"]:
            path = f"sequence_info/{root}/{kind}_vecs/seq.npy"
            if path == missing:
                continue
            a = declare(artifact_identity(e.specification, "KinForm-H", "kcat", path))
            a.checksum = "c" * 64
            a.size = 12
            a.save()
            ArtifactLocation.objects.create(artifact=a, host=host, ready=True)
    binding = declare(
        artifact_identity(
            e.specification,
            "KinForm-H",
            "kcat",
            "pseq2sites/binding_sites_all.tsv",
            context={"sequences": ["seq"]},
        )
    )
    binding.checksum = "d" * 64
    binding.size = 12
    binding.save()
    ArtifactLocation.objects.create(artifact=binding, host=host, ready=True)
    kinform_graph(e, "KinForm-H", "kcat", {"seq": "MAAA"})
    gpu = list(e.stages.filter(kind="embedding"))
    assert len(gpu) == 1 and gpu[0].payload["step"] == "kinform_esmc_residues"
    assert len(gpu[0].payload["outputs"]) == 1
    pool = e.stages.get(payload__step="kinform_pool")
    assert len(pool.payload["outputs"]) == 1 and pool.artifacts.count() == 2


def test_concurrent_method_planning_shares_one_pinned_producer():
    from concurrent.futures import ThreadPoolExecutor

    from django.db import close_old_connections

    left, right = execution("KinForm-H", "raceleft"), execution("KinForm-L", "raceright")

    def plan(e, method):
        close_old_connections()
        try:
            return embedding_stage(
                e,
                method,
                "kcat",
                {"seq": "MAAA"},
                {"seq": {"sequence_info/prot_t5_last/residue_vecs/seq.npy"}},
                {"step": "kinform_t5_residues"},
            )[0][0].pk
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as workers:
        a, b = workers.submit(plan, left, "KinForm-H"), workers.submit(plan, right, "KinForm-L")
        assert a.result() == b.result()
    assert WorkStage.objects.filter(kind="embedding").count() == 1
