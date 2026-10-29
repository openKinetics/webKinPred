from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from types import SimpleNamespace

current = ContextVar("execution_context", default=None)


@dataclass
class StageContext:
    public_id: str
    events: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)
    cache_writes: list = field(default_factory=list)
    cache_namespace: str = ""
    unit_context: dict = field(default_factory=dict)
    select_units: object = None
    unit_batch_size: int = 128
    progress_base: int = 0
    similarity_cache: dict = field(default_factory=dict)
    similarity_recent: OrderedDict = field(default_factory=OrderedDict)
    similarity_writes: list = field(default_factory=list)
    progress: int = 0
    last_flush: float = 0.0

    def emit(self, operation, data):
        self.events.append({"operation": operation, **data})
        if len(self.events) > 2048:
            del self.events[:1024]
        self.progress = max(
            self.progress,
            self.progress_base + int(data.get("predictions_made", data.get("done", 0)) or 0),
        )
        import os
        from pathlib import Path

        from api.execution.artifacts import atomic_json

        if (
            os.environ.get("WEBKINPRED_PROGRESS_FILE")
            and time.monotonic() - self.last_flush >= 0.25
        ):
            atomic_json(Path(os.environ["WEBKINPRED_PROGRESS_FILE"]), {"progress": self.progress})
            self.last_flush = time.monotonic()


class ContextJob(SimpleNamespace):
    predictions_made = 0
    total_predictions = 0

    def save(self, **kwargs):
        current.get().emit(
            "job_progress", {key: getattr(self, key) for key in kwargs.get("update_fields", [])}
        )


def execution_job(public_id):
    context = current.get()
    if context:
        if context.public_id != public_id:
            raise ValueError("Stage job identity mismatch")
        return ContextJob(public_id=public_id)
    from api.models import Job

    return Job.objects.get(public_id=public_id)


def contextual_progress(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        context = current.get()
        if context:
            context.emit(function.__name__, kwargs)
            return None
        return function(*args, **kwargs)

    return wrapped


@contextmanager
def stage_context(context):
    token = current.set(context)
    try:
        yield context
    finally:
        current.reset(token)
