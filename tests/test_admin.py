from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import django
from django.contrib.admin.sites import AdminSite
from django.test import SimpleTestCase
from django.utils import timezone

os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
django.setup()

from api.admin import JobAdmin, _format_duration  # noqa: E402
from api.models import Job  # noqa: E402


class JobAdminTests(SimpleTestCase):
    def setUp(self):
        self.admin = JobAdmin(Job, AdminSite())
        self.now = timezone.now()

    def job(self, **overrides):
        values = {
            "status": "Pending",
            "submission_time": self.now - timedelta(minutes=2),
            "start_time": None,
            "completion_time": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_job_list_includes_direct_kcat_km_and_duration_columns(self):
        self.assertIn("kcat_km_method_display", self.admin.list_display)
        self.assertIn("elapsed_time", self.admin.list_display)
        self.assertIn("time_in_queue", self.admin.list_display)

    def test_elapsed_time_uses_current_time_for_processing_job(self):
        job = self.job(
            status="Processing",
            start_time=self.now - timedelta(hours=1, minutes=2, seconds=3),
        )

        with patch("api.admin.timezone.now", return_value=self.now):
            self.assertEqual(self.admin.elapsed_time(job), "1h 2m 3s")

    def test_elapsed_time_uses_completion_time_for_completed_job(self):
        started = self.now - timedelta(minutes=10)
        job = self.job(
            status="Completed",
            start_time=started,
            completion_time=started + timedelta(minutes=3, seconds=4),
        )

        self.assertEqual(self.admin.elapsed_time(job), "3m 4s")

    def test_elapsed_time_is_blank_for_pending_job(self):
        self.assertEqual(self.admin.elapsed_time(self.job()), "—")

    def test_time_in_queue_uses_submission_time_for_pending_job(self):
        job = self.job(submission_time=self.now - timedelta(days=1, minutes=2, seconds=3))

        with patch("api.admin.timezone.now", return_value=self.now):
            self.assertEqual(self.admin.time_in_queue(job), "1d 0h 2m 3s")

    def test_time_in_queue_is_blank_after_job_starts(self):
        job = self.job(status="Processing", start_time=self.now)

        self.assertEqual(self.admin.time_in_queue(job), "—")

    def test_duration_is_clamped_to_zero(self):
        self.assertEqual(_format_duration(-1), "0s")
