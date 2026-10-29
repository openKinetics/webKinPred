from django.urls import path

from api.execution.metrics import metrics
from api.execution.views import executor

urlpatterns = [
    path("metrics", metrics),
    path("executors/<slug:host_id>/<slug:operation>", executor),
    path("executors/<slug:host_id>/attempts/<uuid:attempt_id>/<slug:operation>", executor),
]
