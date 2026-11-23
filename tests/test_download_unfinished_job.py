import pytest
from django.http import Http404
from django.test import RequestFactory

from api.models import Job
from api.views.file_views import download_job_output


@pytest.mark.django_db
def test_download_of_unfinished_job_is_not_found(settings, tmp_path):
    settings.MEDIA_ROOT = str(tmp_path)
    Job.objects.create(public_id="pending01", status="Processing", prediction_type="kcat")
    with pytest.raises(Http404):
        download_job_output(RequestFactory().get("/api/jobs/pending01/download/"), "pending01")


@pytest.mark.django_db
def test_download_of_completed_job_returns_its_csv(settings, tmp_path):
    settings.MEDIA_ROOT = str(tmp_path)
    (tmp_path / "jobs" / "done0001").mkdir(parents=True)
    (tmp_path / "jobs" / "done0001" / "output.csv").write_text("a,b\n1,2\n")
    Job.objects.create(
        public_id="done0001",
        status="Completed",
        prediction_type="kcat",
        output_file="jobs/done0001/output.csv",
    )
    response = download_job_output(RequestFactory().get("/api/jobs/done0001/download/"), "done0001")
    assert response["Content-Type"] == "text/csv"
    assert b"".join(response.streaming_content) == b"a,b\n1,2\n"
