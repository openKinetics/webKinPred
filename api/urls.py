# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('submit-job/', views.submit_job, name='submit_job'),
    path('job-status/<int:job_id>/', views.job_status, name='job_status'),
]