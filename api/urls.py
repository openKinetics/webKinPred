# api/urls.py
from django.urls import path
from .views import job_views, file_views, health_views, progress_views, similarity_views, validation_views
from .views.csrf_views import get_csrf

urlpatterns = [
    path("health/", health_views.health_check, name="health_check"),
    path("submit-job/", job_views.submit_job, name="submit_job"),
    path("job-status/<public_id>/", job_views.job_status, name="job_status"),
    path(
        "detect-csv-format/",
        file_views.detect_csv_format,
        name="detect_csv_format",
    ),
    path("validate-input/", validation_views.validate_input, name="validate-input"),
    path(
        "cancel-validation/",
        validation_views.cancel_validation,
        name="cancel-validation",
    ),
    path(
        "sequence-similarity-summary/",
        similarity_views.sequence_similarity_summary,
        name="sequence_similarity_summary",
    ),
    path(
        "progress-stream/", progress_views.progress_stream, name="progress-stream"
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
