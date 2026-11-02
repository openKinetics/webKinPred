"""Explicit opt-in. Import last, including after Docker database overrides."""

from __future__ import annotations

import json
import os
from urllib.parse import unquote, urlparse


def configure(namespace):
    enabled = os.getenv("EXECUTION_V2_ENABLED", "0") == "1"
    namespace["EXECUTION_V2_ENABLED"] = enabled
    namespace["EXECUTION_START_PAUSED"] = os.getenv("EXECUTION_START_PAUSED", "1") == "1"
    namespace["EXECUTION_MODEL_VERSIONS"] = json.loads(
        os.getenv("EXECUTION_MODEL_VERSIONS_JSON", "{}")
    )
    namespace["EXECUTION_STAGE_VERSIONS"] = json.loads(
        os.getenv("EXECUTION_STAGE_VERSIONS_JSON", "{}")
    )
    namespace["EXECUTION_BLOB_ROOT"] = os.getenv("EXECUTION_BLOB_ROOT", "/var/lib/webkinpred/blobs")
    namespace["ALLOWED_HOSTS"] = list(namespace["ALLOWED_HOSTS"]) + [host for host in os.getenv("EXECUTION_ALLOWED_HOSTS", "").split(",") if host]
    namespace["EXECUTION_MAINTENANCE_FILE"] = os.getenv(
        "EXECUTION_MAINTENANCE_FILE", "/run/webkinpred/maintenance"
    )
    namespace["EXECUTION_HOSTS"] = json.loads(os.getenv("EXECUTION_HOSTS_JSON", "{}"))
    namespace["EXECUTION_METRICS_TOKEN"] = os.getenv("EXECUTION_METRICS_TOKEN", "")
    namespace["EXECUTION_PREPARED_BACKGROUND_LIMIT"] = int(
        os.getenv("EXECUTION_PREPARED_BACKGROUND_LIMIT", "128")
    )
    namespace["EXECUTION_PREPARED_CPU_LIMIT"] = int(
        os.getenv("EXECUTION_PREPARED_CPU_LIMIT", "32")
    )
    namespace["EXECUTION_PREPARATION_RESERVE_BYTES"] = int(
        os.getenv("EXECUTION_PREPARATION_RESERVE_BYTES", str(2 * 1024**3))
    )
    for alias, variable in (
        ("default", "DATABASE_URL"),
        ("seqmap", "SEQMAP_DATABASE_URL"),
        ("prediction_store", "PREDICTION_STORE_DATABASE_URL"),
    ):
        value = os.getenv(variable)
        if not value:
            continue
        url = urlparse(value)
        if url.scheme not in {"postgres", "postgresql"}:
            raise ValueError(f"{variable} must use PostgreSQL")
        namespace["DATABASES"][alias] = {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": unquote(url.path.lstrip("/")),
            "USER": unquote(url.username or ""),
            "PASSWORD": unquote(url.password or ""),
            "HOST": url.hostname or "",
            "PORT": url.port or 5432,
            "CONN_MAX_AGE": 0,
            "OPTIONS": {"connect_timeout": 5},
        }
    if os.getenv("WEBKINPRED_MEDIA_ROOT"):
        namespace["MEDIA_ROOT"] = os.environ["WEBKINPRED_MEDIA_ROOT"]
    if enabled:
        if namespace["DATABASES"]["default"]["ENGINE"] != "django.db.backends.postgresql":
            raise ValueError("EXECUTION_V2_ENABLED requires PostgreSQL")
        namespace["CELERY_TASK_DEFAULT_QUEUE"] = "webkinpred-control-v2"
        # Redis transport acknowledgement keys are shared by consumers within
        # one logical database, even when their queues have different names.
        # Keep the frozen legacy broker (including unacknowledged deliveries)
        # separate from control tasks throughout migration and rollback.
        legacy_broker = namespace["CELERY_BROKER_URL"]
        broker = os.getenv("EXECUTION_CONTROL_BROKER_URL") or urlparse(
            legacy_broker
        )._replace(path="/3").geturl()
        def broker_database(value):
            parsed = urlparse(value)
            return parsed.hostname, parsed.port or 6379, int(parsed.path.lstrip("/") or "0")

        if (
            urlparse(broker).scheme not in {"redis", "rediss"}
            or broker_database(broker) == broker_database(legacy_broker)
        ):
            raise ValueError("Execution control requires a separate Redis broker database")
        namespace["CELERY_BROKER_URL"] = broker
        namespace["CELERY_RESULT_BACKEND"] = broker
        namespace["CELERY_TASK_ROUTES"] = {"api.tasks.*": {"queue": "webkinpred-control-v2"}}
        namespace["CELERY_TASK_ACKS_LATE"] = True
        namespace["CELERY_WORKER_PREFETCH_MULTIPLIER"] = 1
