from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webKinPred.settings")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.observability.context import bind_log_context, get_log_context, log_context, reset_log_context
from api.observability.formatters import CorrelationFilter, JsonLogFormatter
from api.observability.middleware import RequestIDMiddleware


class ObservabilityFormatterTests(unittest.TestCase):
    def test_context_binding_and_json_formatter(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(CorrelationFilter())
        handler.setFormatter(JsonLogFormatter())

        logger = logging.getLogger("tests.observability.formatter")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        with log_context(job_public_id="job123", celery_task_id="task456", method_key="EITLEM", target="kcat"):
            logger.info("hello", extra={"event": "test.event"})

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["message"], "hello")
        self.assertEqual(payload["event"], "test.event")
        self.assertEqual(payload["job_public_id"], "job123")
        self.assertEqual(payload["celery_task_id"], "task456")
        self.assertEqual(payload["method_key"], "EITLEM")
        self.assertEqual(payload["target"], "kcat")
        self.assertIn("timestamp", payload)

    def test_bind_reset_context(self):
        token = bind_log_context(request_id="req_1")
        try:
            self.assertEqual(get_log_context()["request_id"], "req_1")
        finally:
            reset_log_context(token)
        self.assertNotIn("request_id", get_log_context())

    def test_celery_received_log_infers_run_multi_prediction_context(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(CorrelationFilter())
        handler.setFormatter(JsonLogFormatter())

        logger = logging.getLogger("tests.observability.celery_received")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info(
            "Task api.tasks.run_multi_prediction[task-1] received",
            extra={
                "data": {
                    "id": "task-1",
                    "name": "api.tasks.run_multi_prediction",
                    "args": "('job-1', ['kcat'], {'kcat': 'TurNup'}, {}, True, True, False)",
                    "kwargs": "{}",
                }
            },
        )

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["job_public_id"], "job-1")
        self.assertEqual(payload["celery_task_id"], "task-1")
        self.assertEqual(payload["method_key"], "TurNup")
        self.assertEqual(payload["target"], "kcat")


class RequestIDMiddlewareTests(unittest.TestCase):
    def test_generates_and_returns_request_id(self):
        request = SimpleNamespace(META={})

        def get_response(req):
            self.assertTrue(req.request_id)
            return {}

        response = RequestIDMiddleware(get_response)(request)
        self.assertEqual(response["X-Request-ID"], request.request_id)

    def test_propagates_existing_request_id(self):
        request = SimpleNamespace(META={"HTTP_X_REQUEST_ID": "req-existing"})
        response = RequestIDMiddleware(lambda _req: {})(request)
        self.assertEqual(response["X-Request-ID"], "req-existing")


class CelerySignalLoggingTests(unittest.TestCase):
    def test_run_multi_prediction_context_from_args(self):
        from api.observability import celery_signals

        task = SimpleNamespace(name="api.tasks.run_multi_prediction", request=SimpleNamespace())
        context = celery_signals._task_context(
            task,
            "task-1",
            ("job-1", ["kcat"], {"kcat": "TurNup"}, {}, True, True, False),
            {},
        )

        self.assertEqual(context["job_public_id"], "job-1")
        self.assertEqual(context["celery_task_id"], "task-1")
        self.assertEqual(context["method_key"], "TurNup")
        self.assertEqual(context["target"], "kcat")

    def test_run_multi_prediction_context_compacts_multiple_targets(self):
        from api.observability import celery_signals

        task = SimpleNamespace(name="api.tasks.run_multi_prediction", request=SimpleNamespace())
        context = celery_signals._task_context(
            task,
            "task-1",
            ("job-1", ["kcat", "Km"], {"kcat": "TurNup", "Km": "UniKP"}),
            {},
        )

        self.assertEqual(context["method_key"], "TurNup,UniKP")
        self.assertEqual(context["target"], "kcat,Km")

    def test_task_lifecycle_logs_context(self):
        from api.observability import celery_signals

        task = SimpleNamespace(name="api.tasks.run_prediction", request=SimpleNamespace())
        with self.assertLogs("api.observability.celery", level="INFO") as logs:
            celery_signals.log_task_prerun(
                task_id="task-1",
                task=task,
                args=("job-1", "DLKcat", "kcat"),
                kwargs={},
            )
            celery_signals.log_task_postrun(
                task_id="task-1",
                task=task,
                args=("job-1", "DLKcat", "kcat"),
                kwargs={},
                state="SUCCESS",
            )

        self.assertIn("Celery task started", logs.output[0])
        self.assertIn("Celery task finished", logs.output[1])


class SubprocessRunnerLoggingTests(unittest.TestCase):
    def _import_runner_or_skip(self):
        try:
            from api.prediction_engines import subprocess_runner as runner
        except ModuleNotFoundError as exc:
            if exc.name == "django":
                self.skipTest("Django is not installed in this Python environment.")
            raise
        return runner

    def test_progress_updates_and_no_raw_print(self):
        runner = self._import_runner_or_skip()

        job = SimpleNamespace(public_id="job-sub", predictions_made=0, total_predictions=0)
        job.save = lambda update_fields=None: None
        progress_calls: list[tuple[int, int | None]] = []

        code = (
            "import sys\n"
            "print('MoleculeModel(')\n"
            "print('Progress: 1/3')\n"
            "print('ordinary verbose model line')\n"
            "print('Progress: 3/3')\n"
        )

        with patch.object(runner, "start_embedding_tracking", return_value=False):
            with patch.object(runner, "set_stage_prediction_progress") as set_progress:
                set_progress.side_effect = lambda **kw: progress_calls.append((kw["done"], kw.get("total")))
                with self.assertLogs("api.prediction_engines.subprocess_runner", level="INFO") as logs:
                    runner.run_prediction_subprocess(
                        command=[sys.executable, "-c", code],
                        job=job,
                        label="CatPred",
                        method_key="CatPred",
                        target="kcat",
                        valid_sequences=["A", "B", "C"],
                    )

        self.assertEqual(progress_calls, [(1, 3), (3, 3)])
        joined = "\n".join(logs.output)
        self.assertIn("Starting prediction subprocess", joined)
        self.assertIn("Subprocess prediction progress", joined)
        self.assertIn("Prediction subprocess completed", joined)
        self.assertNotIn("MoleculeModel(", joined)

    def test_failed_subprocess_logs_return_code(self):
        runner = self._import_runner_or_skip()

        job = SimpleNamespace(public_id="job-fail", predictions_made=0, total_predictions=0)
        job.save = lambda update_fields=None: None

        with patch.object(runner, "start_embedding_tracking", return_value=False):
            with self.assertLogs("api.prediction_engines.subprocess_runner", level="ERROR") as logs:
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run_prediction_subprocess(
                        command=[sys.executable, "-c", "import sys; sys.exit(7)"],
                        job=job,
                        label="DLKcat",
                        method_key="DLKcat",
                        target="kcat",
                        valid_sequences=["A"],
                    )

        self.assertIn("Prediction subprocess failed", "\n".join(logs.output))


class NoBarePrintGuardTests(unittest.TestCase):
    def test_no_bare_print_in_api_runtime_paths(self):
        root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for path in (root / "api").rglob("*.py"):
            if "/migrations/" in path.as_posix():
                continue
            import ast
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "print" for node in ast.walk(tree)):
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
