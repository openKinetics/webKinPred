import base64
import json

import pytest
from tools.executor.migration import decode_message, reconcile


def message(ident="job1", experimental=None):
    body = [
        [ident, ["kcat"], {"kcat": "DLKcat"}, experimental or {}, False, False, False, True],
        {},
        {},
    ]
    return json.dumps(
        {
            "headers": {"task": "api.tasks.run_multi_prediction", "id": "task1"},
            "properties": {"body_encoding": "base64"},
            "body": base64.b64encode(json.dumps(body).encode()).decode(),
        }
    ).encode()


def test_export_preserves_nondefault_options_and_experimental_overrides():
    override = {"kcat": [{"idx": 1, "found": True, "kcat_value": 4.2}]}
    request = decode_message(message(experimental=override))
    assert request["canonicalize_substrates"] is False
    assert request["include_similarity_columns"] is False
    assert request["recon_xkg"] is True
    assert request["experimental_results"] == override


def test_reconcile_deduplicates_delivery_but_preserves_processing_identity():
    request = decode_message(message())
    result = reconcile([{"public_id": "job1", "status": "Processing"}], [request, request])
    assert result["ready"]
    assert len(result["specifications"]) == 1
    assert result["dispositions"][0]["disposition"] == "restart"
    assert result["dispositions"][0]["duplicate_deliveries"] == 1


def test_reconcile_never_guesses_missing_or_conflicting_options():
    jobs = [{"public_id": "job1", "status": "Pending"}]
    assert not reconcile(jobs, [])["ready"]
    request = decode_message(message())
    assert not reconcile(jobs, [request, {**request, "include_similarity_columns": True}])["ready"]
    assert not reconcile(jobs, [None])["ready"]


def test_compressed_requests_are_preserved_for_review_not_silently_changed():
    envelope = json.loads(message())
    envelope["headers"]["compression"] = "gzip"
    with pytest.raises(ValueError, match="serialization"):
        decode_message(json.dumps(envelope))


@pytest.mark.django_db(transaction=True)
def test_final_export_reuses_captured_requests_after_workers_stop(tmp_path, monkeypatch, settings):
    from types import SimpleNamespace

    from api.management.commands import execution_export
    from api.models import Job
    from django.core.management import call_command

    settings.CELERY_BROKER_URL = "redis://isolated/0"
    Job.objects.create(public_id="job1", status="Processing", prediction_type="kcat")
    captured = tmp_path / "workers.json"
    captured.write_text(
        json.dumps(
            {
                "active": {
                    "frozen-worker": [
                        {
                            "name": "api.tasks.run_multi_prediction",
                            "id": "original-task",
                            "args": [
                                "job1",
                                ["kcat"],
                                {"kcat": "DLKcat"},
                                {},
                                False,
                                False,
                                False,
                                True,
                            ],
                            "kwargs": {},
                        }
                    ]
                },
                "reserved": {"frozen-worker": []},
                "scheduled": {"frozen-worker": []},
            }
        )
    )
    monkeypatch.setattr("redis.Redis.from_url", lambda *a, **k: object())
    monkeypatch.setattr(
        "django_redis.get_redis_connection", lambda *a: SimpleNamespace(scan_iter=lambda **k: [])
    )
    monkeypatch.setattr(execution_export, "export_broker", lambda *a: [])
    monkeypatch.setattr(
        execution_export.app.control,
        "inspect",
        lambda **k: pytest.fail("Stopped workers must not be inspected"),
    )
    destination = tmp_path / "final"
    call_command("execution_export", str(destination), workers_from=captured, writers_quiesced=True)
    manifest = json.loads((destination / "reconciliation.json").read_text())
    assert manifest["ready"] and manifest["unfinished_count"] == 1
    assert manifest["dispositions"][0]["disposition"] == "restart"
    assert manifest["specifications"]["job1"]["canonicalize_substrates"] is False
    assert (destination / "worker-capture-provenance.json").is_file()


def test_legacy_gpu_export_preserves_hex_ids_options_and_sequence_collisions(tmp_path, monkeypatch):
    import io
    import sqlite3

    from tools.executor.export_legacy_gpu import export

    ident = "123456781234123412341234567890ab"
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / (ident + ".log")).touch()
    database = tmp_path / "sequences.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE sequences (id TEXT PRIMARY KEY, seq TEXT)")
        db.execute("INSERT INTO sequences VALUES (?, ?)", ["abc_2", "MAAAA"])
    responses = {
        "/health": {"active_jobs": 0, "queued_jobs": 1},
        "/embed/jobs/" + ident: {
            "job_id": ident,
            "status": "queued",
            "method_key": "CatPred",
            "target": "Km",
            "profile": "historical-profile",
            "step_work": {"catpred_embed_km": ["abc_2"]},
        },
    }

    def open_url(request, timeout):
        return io.BytesIO(
            json.dumps(responses[request.full_url.removeprefix("http://gpu")]).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    result = export("http://gpu", "secret", logs, database, tmp_path / "export")
    assert result["ready"]
    request = result["unfinished_requests"][0]
    assert request["legacy_job_id"] == ident
    assert request["request"]["seq_id_to_seq"] == {"abc_2": "MAAAA"}
    assert request["request"]["profile"] == "historical-profile"
    assert "secret" not in (tmp_path / "export/gpu-requests.json").read_text()
    responses["/health"]["queued_jobs"] = 2
    with pytest.raises(RuntimeError, match="logs are incomplete"):
        export("http://gpu", "secret", logs, database, tmp_path / "incomplete-export")
