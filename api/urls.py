# api/urls.py
from django.urls import path
from .views import pred_jobs_views,valid_inputs_views
urlpatterns = [
    path('submit-job/', pred_jobs_views.submit_job, name='submit_job'),
    path('job-status/<public_id>/', pred_jobs_views.job_status, name='job_status'),
    path('detect-csv-format/', pred_jobs_views.detect_csv_format, name='detect_csv_format'),
    path('validate-input/', valid_inputs_views.validate_input, name='validate-input'),
    path('sequence-similarity-summary/', valid_inputs_views.sequence_similarity_summary, name='sequence_similarity_summary'),
    path('progress-stream/', valid_inputs_views.progress_stream, name='progress-stream'),
    path('cancel-validation/', valid_inputs_views.cancel_validation, name='cancel-validation'),
]