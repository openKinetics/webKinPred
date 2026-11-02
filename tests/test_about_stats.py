import csv
import json
from datetime import timedelta
from unittest.mock import patch

import pytest
from api.models import AboutJobProtein, AboutJobStats, Job
from api.services import about_stats_service as stats
from api.views.stats_views import about_stats
from django.core.cache import cache
from django.test import RequestFactory
from django.utils import timezone

pytestmark = pytest.mark.django_db(transaction=True)


def result(tmp_path, name="result", rows=None, fields=None):
    path = tmp_path / (name + ".csv")
    rows = rows or []
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def completed(path, name="statsjob"):
    return Job.objects.create(
        public_id=name,
        status="Completed",
        prediction_type="",
        completion_time=timezone.now(),
        output_file=str(path),
    )


def publish():
    stats.sync_completed_jobs(limit=200)
    return stats.refresh_about_stats_cache(force=True)


@pytest.fixture(autouse=True)
def clear_cache(settings, tmp_path):
    settings.MEDIA_ROOT = str(tmp_path)
    cache.clear()
    yield
    cache.clear()


def test_current_headers_ranges_arrays_invalid_values_and_successful_proteins(tmp_path):
    path = result(
        tmp_path,
        rows=[
            {"Protein Sequence": "AAA", "kcat (1/s)": "1", "KM (mM)": "[2,null,3]"},
            {
                "Protein Sequence": "BBB",
                "Predicted kcat range: kcat (1/s)": "1 to 10 s^-1",
                "Predicted KM range: kM (M)": '["1e-05 to 0.0001 M",null]',
            },
            {"Protein Sequence": "CCC", "kcat/Km (1/(s·M))": "4"},
            {"Protein Sequence": "DDD", "kcat/Km (1/(s*mM))": "5"},
            {"Protein Sequence": "FAILED", "kcat (1/s)": "NaN", "KM (mM)": "[null,NaN]"},
            {"Protein Sequence": "INVALID", "kcat (1/s)": "error", "KM (mM)": "inf"},
            {"Protein Sequence": "", "kcat (1/s)": "0"},
        ],
    )
    completed(path)
    payload = publish()
    assert payload["jobs_completed"] == 1
    assert payload["reactions_completed"] == 5
    assert payload["unique_protein_sequences"] == 4
    assert payload["kcat_predictions_completed"] == 3
    assert payload["km_predictions_completed"] == 3
    assert payload["kcat_km_predictions_completed"] == 2
    assert payload["parameter_predictions_completed"] == 8


def test_summaries_are_incremental_and_duplicate_delivery_cannot_double_count(tmp_path):
    first = completed(
        result(tmp_path, "one", [{"Protein Sequence": "AAA", "kcat (1/s)": "1"}]), "one"
    )
    assert publish()["jobs_completed"] == 1
    completed(
        result(
            tmp_path,
            "two",
            [
                {"Protein Sequence": "AAA", "kcat (1/s)": "2"},
                {"Protein Sequence": "BBB", "kcat (1/s)": "3"},
            ],
        ),
        "two",
    )
    with patch.object(stats, "summarise_output", wraps=stats.summarise_output) as parse:
        payload = publish()
        assert parse.call_count == 1
        assert stats.sync_completed_jobs() == 0
        assert parse.call_count == 1
    assert payload["jobs_completed"] == 2
    assert payload["parameter_predictions_completed"] == 3
    assert payload["unique_protein_sequences"] == 2
    assert AboutJobStats.objects.filter(job=first).count() == 1


def test_changed_result_replaces_counts_and_memberships(tmp_path):
    path = result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}])
    job = completed(path)
    publish()
    result(tmp_path, rows=[{"Protein Sequence": "BBB", "KM (mM)": "[1,2]"}])
    Job.objects.filter(pk=job.pk).update(completion_time=timezone.now())
    payload = publish()
    assert payload["jobs_completed"] == 1
    assert payload["kcat_predictions_completed"] == 0
    assert payload["km_predictions_completed"] == 2
    assert payload["unique_protein_sequences"] == 1
    assert AboutJobProtein.objects.count() == 1


def test_noncompleted_and_deleted_jobs_are_excluded_with_shared_proteins_preserved(tmp_path):
    path = result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}])
    a, b = completed(path, "one"), completed(path, "two")
    publish()
    Job.objects.filter(pk=a.pk).update(status="Processing")
    payload = stats.refresh_about_stats_cache(force=True)
    assert payload["jobs_completed"] == payload["unique_protein_sequences"] == 1
    b.delete()
    payload = stats.refresh_about_stats_cache(force=True)
    assert payload["jobs_completed"] == payload["unique_protein_sequences"] == 0
    assert AboutJobProtein.objects.count() == 1  # Processing job remains safely re-countable.


def test_completion_during_file_read_is_rechecked_before_commit(tmp_path):
    path = result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}])
    job = completed(path)
    real = stats.summarise_output

    def changed(p, **kwargs):
        Job.objects.filter(pk=job.pk).update(status="Processing")
        return real(p, **kwargs)

    with patch.object(stats, "summarise_output", side_effect=changed):
        stats.sync_completed_jobs()
    assert not AboutJobStats.objects.exists()


def test_missing_results_are_visible_and_retry_does_not_starve_other_jobs(tmp_path):
    missing = completed(tmp_path / "missing.csv", "missing")
    completed(result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}]))
    payload = publish()
    assert payload["jobs_completed"] == 2
    assert payload["jobs_with_unavailable_results"] == 1
    assert payload["parameter_predictions_completed"] == 1
    assert stats.sync_completed_jobs() == 0
    missing.refresh_from_db()
    assert missing.status == "Completed"


def test_partial_backfill_is_not_published_and_work_is_bounded(tmp_path):
    path = result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}])
    completed(path, "one")
    completed(path, "two")
    assert stats.sync_completed_jobs(limit=1) == 1
    assert stats.refresh_about_stats_cache(force=True) is None
    assert stats.sync_completed_jobs(limit=1) == 1
    assert stats.refresh_about_stats_cache(force=True)["jobs_completed"] == 2


def test_multisequence_metadata_counts_successful_candidates_not_failed_ones(tmp_path):
    extra = json.dumps(
        [
            {"sequenceIndex": 1, "sequence": "AAA", "prediction": 1, "error": None},
            {"sequenceIndex": 2, "sequence": "BBB", "prediction": None, "error": "invalid"},
        ]
    )
    completed(
        result(
            tmp_path,
            rows=[{"Protein Sequence": "AAA;BBB", "kcat (1/s)": "1", "Extra Info kcat": extra}],
        )
    )
    assert publish()["unique_protein_sequences"] == 1


def test_legacy_null_completion_times_and_bom_are_supported(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("\ufeffProtein Sequence,kcat (1/s)\nAAA,1\n")
    job = completed(path)
    Job.objects.filter(pk=job.pk).update(completion_time=None)
    assert publish()["jobs_completed"] == 1
    assert stats.sync_completed_jobs() == 0


def test_http_reads_no_files_or_aggregates_and_supports_etag(tmp_path, django_assert_num_queries):
    completed(result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}]))
    payload = publish()
    with (
        patch.object(stats, "summarise_output", side_effect=AssertionError("HTTP cannot read CSV")),
        patch.object(
            stats, "refresh_about_stats_cache", side_effect=AssertionError("HTTP cannot aggregate")
        ),
    ):
        with django_assert_num_queries(1):
            assert stats.get_about_stats() == payload
        with django_assert_num_queries(0):
            assert stats.get_about_stats() == payload
        response = about_stats(RequestFactory().get("/api/about-stats/"))
        assert response.status_code == 200
        assert "max-age=60" in response["Cache-Control"]
        again = about_stats(
            RequestFactory().get("/api/about-stats/", HTTP_IF_NONE_MATCH=response["ETag"])
        )
        assert again.status_code == 304


def test_expired_payload_is_never_served_indefinitely(tmp_path):
    completed(result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}]))
    publish()
    future = timezone.now() + timedelta(minutes=21)
    with patch.object(stats.timezone, "now", return_value=future):
        response = about_stats(RequestFactory().get("/api/about-stats/"))
    assert response.status_code == 503
    assert response["Cache-Control"] == "no-store"
    assert response["Retry-After"] == "60"


def test_missing_snapshot_does_not_trigger_an_expensive_request_time_rebuild():
    with patch.object(stats, "summarise_output", side_effect=AssertionError("no scan")):
        assert about_stats(RequestFactory().get("/api/about-stats/")).status_code == 503


def test_nested_substrate_metadata_retains_parent_protein_identity(tmp_path):
    extra = json.dumps(
        [
            {"sequenceIndex": 1, "sequence": "AAA", "substrates": [{"prediction": 1}]},
            {
                "sequenceIndex": 2,
                "sequence": "BBB",
                "substrates": [{"prediction": None, "error": "bad"}],
            },
        ]
    )
    completed(
        result(
            tmp_path,
            rows=[{"Protein Sequence": "AAA;BBB", "KM (mM)": "[1,null]", "Extra Info Km": extra}],
        )
    )
    assert publish()["unique_protein_sequences"] == 1


def test_restart_reusing_a_public_id_does_not_double_count(tmp_path):
    path = result(tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1"}])
    job = completed(path)
    publish()
    job.delete()
    completed(path)
    payload = publish()
    assert payload["jobs_completed"] == payload["parameter_predictions_completed"] == 1


def test_legacy_completion_refresh_only_marks_dirty_without_reading_files():
    from api.models import AboutStatsCache
    from api.tasks import _safe_refresh_about_stats_cache

    row = AboutStatsCache.objects.create(key=stats.CACHE_KEY, is_stale=False)
    with patch.object(stats, "summarise_output", side_effect=AssertionError("No scan")):
        _safe_refresh_about_stats_cache()
    row.refresh_from_db()
    assert row.is_stale


def test_legacy_realkcat_units_and_requested_targets_ignore_input_measurements(tmp_path):
    path = result(
        tmp_path,
        rows=[
            {
                "Protein Sequence": "AAA",
                "Km (M)": "2",
                "KM (mM)": "[100,200,300]",
                "kcat (1/s)": "999",
            }
        ],
    )
    job = completed(path)
    Job.objects.filter(pk=job.pk).update(prediction_type="Km", km_method="RealKcat")
    payload = publish()
    assert payload["km_predictions_completed"] == 1
    assert payload["kcat_predictions_completed"] == 0


def test_unrequested_parameter_columns_do_not_count_as_predictions(tmp_path):
    path = result(
        tmp_path, rows=[{"Protein Sequence": "AAA", "kcat (1/s)": "1", "KM (mM)": "[10,20]"}]
    )
    job = completed(path)
    Job.objects.filter(pk=job.pk).update(prediction_type="kcat", kcat_method="DLKcat")
    payload = publish()
    assert payload["parameter_predictions_completed"] == 1
