# webKinPred/celery.py

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")

app = Celery("webKinPred")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

app.conf.task_default_queue = (
    "webkinpred-control-v2" if os.getenv("EXECUTION_V2_ENABLED") == "1" else "webkinpred"
)
app.conf.worker_hijack_root_logger = False
app.conf.worker_redirect_stdouts = False
