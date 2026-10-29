import pytest
from api.execution.embedding_api import dispatch, status, submit
from api.execution.models import ArtifactLocation, EmbeddingSubmission, ExecutorHost
from api.execution.planning import sequence_id
from django.test import override_settings


@pytest.mark.django_db(transaction=True)
def test_shared_and_cached_embedding_submissions_follow_the_actual_producer():
    host = ExecutorHost.objects.create(id="gpu")
    sid = sequence_id("MAAA")
    payload = {"method_key": "UniKP", "target": "kcat", "seq_id_to_seq": {sid: "MAAA"}}
    with override_settings(EXECUTION_MODEL_VERSIONS={"UniKP": {"weights": "pinned"}}):
        first = submit({**payload, "request_id": "first"})
        second = submit({**payload, "request_id": "second"})
        assert submit({**payload, "request_id": "first"})["job_id"] == first["job_id"]
        original = EmbeddingSubmission.objects.get(pk=first["job_id"])
        shared = EmbeddingSubmission.objects.get(pk=second["job_id"])
        assert not shared.execution.stages.filter(kind="embedding").exists()
        producer = original.execution.stages.get(kind="embedding")
        producer.state = "running"
        producer.save()
        assert status(shared)["status"] != "done"
        producer.state = "completed"
        producer.save()
        # Completion metadata alone is insufficient: verified artifacts must exist.
        assert status(shared)["status"] != "done"
        artifact = producer.outputs.get()
        ArtifactLocation.objects.create(artifact=artifact, host=host, ready=True)
        assert status(shared)["status"] == "done"
        third = submit({**payload, "request_id": "already-cached"})
        cached = EmbeddingSubmission.objects.get(pk=third["job_id"])
        assert status(cached)["status"] == "done"
        assert dispatch("embed-health", {}) == {"active_jobs": 0, "queued_jobs": 0}
