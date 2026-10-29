from types import SimpleNamespace

import pytest
from api.execution.artifacts import sha256_file
from api.execution.coverage import Coverage
from api.execution.shapes import prediction_shape


def request(tmp_path, method="DLKcat"):
    source = tmp_path / "jobs/frozen/input.csv"
    source.parent.mkdir(parents=True)
    source.write_text("Protein Sequence,Substrate\nMAAA;MCCC,CC\n")
    return SimpleNamespace(
        job=SimpleNamespace(public_id="frozen"),
        checksum="frozen-specification",
        input_checksum=sha256_file(source),
        specification={
            "targets": ["kcat"],
            "methods": {"kcat": method},
            "model_versions": {method: {"weights": "pinned"}},
            "execution_versions": {"assemble": {"implementation": "pinned"}},
            "handle_long_sequences": "truncate",
            "canonicalize_substrates": True,
            "include_similarity_columns": False,
        },
    )


def profile(kind, method, identity, limits):
    return SimpleNamespace(
        stage_kind=kind,
        method=method,
        fingerprint=kind,
        settings={"model_identity": identity},
        limits=limits,
        evidence={"runtime": {"hosts": {"production": "host-fingerprint"}}},
    )


@pytest.mark.django_db(transaction=True)
def test_deferred_coverage_checks_inputs_without_creating_artifacts_sequences_or_quota(tmp_path):
    from api.execution.models import Artifact, QuotaEntry, SequenceIdentity, WorkStage

    execution = request(tmp_path)
    auditor = Coverage(
        [
            profile(
                "predict",
                "DLKcat",
                {"weights": "pinned"},
                {"rows": 128, "units": 256, "max_length": 10000},
            ),
            profile("assemble", "", {"implementation": "pinned"}, {"rows": 10000}),
        ]
    )
    result = auditor.execution(execution, tmp_path)
    assert result == {
        "rows": 1,
        "input": execution.input_checksum,
        "specification": execution.checksum,
    }
    assert auditor.used == {"predict", "assemble"}
    assert all(
        model.objects.count() == 0 for model in [Artifact, QuotaEntry, SequenceIdentity, WorkStage]
    )
    execution.input_checksum = "changed"
    with pytest.raises(ValueError, match="checksum"):
        auditor.execution(execution, tmp_path)


def test_deferred_coverage_cannot_assume_or_fallback_for_missing_gpu_profiles(tmp_path):
    execution = request(tmp_path, "UniKP")
    auditor = Coverage(
        [profile("predict", "UniKP", {"weights": "pinned"}, {"rows": 128, "units": 256})]
    )
    with pytest.raises(ValueError, match="embedding/UniKP"):
        auditor.execution(execution, tmp_path)


def test_turnup_reserves_normalized_model_length_and_preserves_raw_input_bytes():
    batch = SimpleNamespace(sequences=["M" * 1024], call_kwargs={"substrates": ["CC"]})
    assert prediction_shape([None], batch, 5000, "TurNup") == {
        "rows": 1,
        "units": 1,
        "max_length": 1022,
        "max_molecule_length": 2,
        "input_bytes": 5000,
    }
