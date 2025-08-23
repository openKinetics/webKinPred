# api/urls.py
from django.urls import path
from .views import job_views, file_views, health_views, valid_inputs_views
from .csrf import get_csrf


urlpatterns = [
    path("health/", health_views.health_check, name="health_check"),
    path("submit-job/", job_views.submit_job, name="submit_job"),
    path("job-status/<public_id>/", job_views.job_status, name="job_status"),
    path(
        "detect-csv-format/",
        file_views.detect_csv_format,
        name="detect_csv_format",
    ),
    path("validate-input/", valid_inputs_views.validate_input, name="validate-input"),
    path(
        "sequence-similarity-summary/",
        valid_inputs_views.sequence_similarity_summary,
        name="sequence_similarity_summary",
    ),
    path(
        "progress-stream/", valid_inputs_views.progress_stream, name="progress-stream"
    ),
    path(
        "cancel-validation/",
        valid_inputs_views.cancel_validation,
        name="cancel-validation",
    ),
    path("csrf/", get_csrf, name="get-csrf"),
    path(
        "jobs/<slug:public_id>/download/",
        file_views.download_job_output,
        name="download_job_output",
    ),
    path(
        "jobs/<slug:public_id>/input/",
        file_views.download_job_input,
        name="download_job_input",
    ),
]
