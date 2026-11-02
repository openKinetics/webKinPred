"""Incremental result summaries and bounded, background-published usage snapshots.

HTTP requests never open result files or count the protein membership table.
Only the singleton worker parses new/changed completed jobs, once per result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from api.models import AboutJobProtein, AboutJobStats, AboutStatsCache, Job
from django.core.cache import cache
from django.core.exceptions import SuspiciousFileOperation
from django.db import connection, transaction
from django.db.models import Count, F, Q, Sum
from django.utils import timezone

log = logging.getLogger(__name__)
CACHE_KEY = "about_stats_v3"
MEMORY_KEY = "about_stats_payload_v3"
VERSION = 3
MAX_AGE = 20 * 60
COUNTS = (
    "jobs_completed",
    "reactions_completed",
    "unique_protein_sequences",
    "parameter_predictions_completed",
    "kcat_predictions_completed",
    "km_predictions_completed",
    "kcat_km_predictions_completed",
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
_RANGE = re.compile(rf"^({_NUMBER})\s+to\s+({_NUMBER})\s+(?:s\^-1|M)$", re.I)


class StatsNotReady(Exception):
    pass


@lru_cache(maxsize=1)
def _columns():
    from api.methods.registry import all_methods

    columns = {
        "kcat": {"kcat (1/s)"},
        "Km": {"KM (mM)", "Km (M)"},
        "kcat/Km": {"kcat/Km (1/(s*mM))", "kcat/Km (1/(s·M))"},
    }
    for descriptor in all_methods().values():
        for target, column in descriptor.output_cols.items():
            columns[target].add(column)
    return columns


def _value_count(value):
    """Count returned parameter values; a CatRange interval is one estimate."""
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_value_count(item) for item in value)
    text = str(value).strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except (ValueError, RecursionError):
            return 0
        return _value_count(decoded) if isinstance(decoded, list) else 0
    match = _RANGE.fullmatch(text)
    if match:
        return int(all(math.isfinite(float(v)) for v in match.groups()))
    try:
        return int(bool(text) and math.isfinite(float(text)))
    except (ValueError, OverflowError):
        return 0


def _sequence_hashes(value):
    for sequence in str(value or "").split(";"):
        sequence = sequence.strip()
        if sequence and sequence.lower() not in {"nan", "none", "null", "<na>"}:
            yield hashlib.sha256(sequence.encode()).digest()


def summarise_output(path, *, methods=None):
    """Stream one CSV; retain only distinct sequence hashes, not its rows."""
    counts = dict(reactions=0, kcat=0, km=0, kcat_km=0)
    proteins = set()
    columns = _columns()
    csv.field_size_limit(8 * 1024**2)
    before = path.stat()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        if "Protein Sequence" not in header or not header.intersection(
            set.union(*columns.values())
        ):
            raise ValueError("Result CSV has no recognised prediction schema")
        from api.methods.registry import all_methods

        selected = {}
        for target, names in columns.items():
            if methods is not None and target not in methods:
                selected[target] = []
                continue
            descriptor = all_methods().get((methods or {}).get(target))
            preferred = descriptor.output_cols.get(target) if descriptor else None
            if (methods or {}).get(target) == "RealKcat" and target == "Km":
                preferred = "Km (M)"
            selected[target] = [preferred] if preferred in header else sorted(header & names)
        if not any(selected.values()):
            raise ValueError("Result CSV has no columns for its requested targets")
        for row in reader:
            # A row/target is represented by one method column. Do not double count aliases.
            values = {
                target: max((_value_count(row.get(c)) for c in names), default=0)
                for target, names in selected.items()
            }
            counts["kcat"] += values["kcat"]
            counts["km"] += values["Km"]
            counts["kcat_km"] += values["kcat/Km"]
            if any(values.values()):
                counts["reactions"] += 1
                # Multi-sequence metadata identifies successful candidates, including
                # non-selected successful candidates, while excluding failed ones.
                found = set()
                has_sequence_metadata = False
                for target, number in (
                    values.items() if ";" in str(row.get("Protein Sequence", "")) else ()
                ):
                    if not number:
                        continue
                    try:
                        metadata = json.loads(row.get("Extra Info " + target) or "null")
                    except (ValueError, TypeError):
                        continue
                    stack = [(metadata, None)]
                    while stack:
                        item, sequence = stack.pop()
                        if isinstance(item, list):
                            stack.extend((value, sequence) for value in item)
                        elif isinstance(item, dict):
                            if "sequence" in item and "sequenceIndex" in item:
                                has_sequence_metadata = True
                                sequence = item["sequence"]
                            if (
                                sequence
                                and not item.get("error")
                                and _value_count(item.get("prediction"))
                            ):
                                found.update(_sequence_hashes(sequence))
                            stack.extend(
                                (value, sequence)
                                for value in item.values()
                                if isinstance(value, (dict, list))
                            )
                proteins.update(
                    found
                    if has_sequence_metadata
                    else _sequence_hashes(row.get("Protein Sequence"))
                )
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError("Result changed while being summarised")
    return counts, proteins


def _pending_jobs():
    same_completion = Q(about_stats__completion_time=F("completion_time")) | Q(
        completion_time__isnull=True, about_stats__completion_time__isnull=True
    )
    current = (
        Q(about_stats__version=VERSION, about_stats__output_name=F("output_file")) & same_completion
    )
    return Job.objects.filter(status="Completed").exclude(current)


def sync_completed_jobs(*, limit=20):
    """Bounded reconciliation also catches lost completion notifications and retries."""
    jobs = list(
        _pending_jobs()
        .exclude(about_stats__retry_after__gt=timezone.now())
        .order_by("pk")
        .only(
            "output_file",
            "completion_time",
            "prediction_type",
            "kcat_method",
            "km_method",
            "kcat_km_method",
        )[: max(1, min(limit, 200))]
    )
    for job in jobs:
        signature = (job.completion_time, job.output_file.name)
        try:
            fields = {"kcat": job.kcat_method, "Km": job.km_method, "kcat/Km": job.kcat_km_method}
            targets = job.prediction_type.split("+")
            methods = (
                {t: fields[t] for t in targets}
                if targets and all(t in fields for t in targets)
                else {t: method for t, method in fields.items() if method}
            )
            counts, proteins = summarise_output(Path(job.output_file.path), methods=methods or None)
            error = ""
        except (OSError, ValueError, csv.Error, UnicodeError, SuspiciousFileOperation) as exc:
            counts, proteins = {}, set()
            error = type(exc).__name__ + ": result unavailable or unreadable"
            log.warning("About result summary unavailable", extra={"job_id": job.pk})
        with transaction.atomic():
            current = Job.objects.select_for_update().filter(pk=job.pk, status="Completed").first()
            if current is None or (current.completion_time, current.output_file.name) != signature:
                continue
            summary, _ = AboutJobStats.objects.get_or_create(job=current)
            # Another reconciler may already have committed the same result.
            if (
                summary.version == VERSION
                and (summary.completion_time, summary.output_name) == signature
            ):
                continue
            summary.proteins.all().delete()
            summary.completion_time, summary.output_name = signature
            summary.version = 0 if error else VERSION
            summary.error = error
            summary.retry_after = timezone.now() + timedelta(minutes=20) if error else None
            for key in ("reactions", "kcat", "km", "kcat_km"):
                setattr(summary, key, counts.get(key, 0))
            summary.save()
            hashes = sorted(proteins)
            for offset in range(0, len(hashes), 1000):
                AboutJobProtein.objects.bulk_create(
                    [
                        AboutJobProtein(summary=summary, sequence_sha256=h)
                        for h in hashes[offset : offset + 1000]
                    ],
                    batch_size=1000,
                )
    return len(jobs)


def mark_about_stats_cache_stale():
    """Cheap hint only; periodic reconciliation is authoritative."""
    try:
        AboutStatsCache.objects.filter(key=CACHE_KEY, is_stale=False).update(is_stale=True)
    except Exception:
        log.warning("About statistics invalidation deferred", exc_info=True)


def refresh_about_stats_cache(*, force=False):
    """Consistent snapshot of summaries; no files are opened here."""
    with transaction.atomic():
        if connection.vendor == "postgresql" and len(connection.atomic_blocks) == 1:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        return _publish_snapshot(force=force)


def _publish_snapshot(*, force=False):
    row, _ = AboutStatsCache.objects.get_or_create(key=CACHE_KEY)
    now = timezone.now()
    if not force and row.generated_at and now - row.generated_at < timedelta(minutes=5):
        return _deserialise_payload(row.payload)
    pending = _pending_jobs()
    unprocessed = pending.filter(Q(about_stats__isnull=True) | Q(about_stats__error="")).exists()
    if unprocessed:
        # Do not publish an apparently complete, partially backfilled snapshot.
        return None
    summaries = AboutJobStats.objects.filter(job__status="Completed", version=VERSION)
    totals = summaries.aggregate(
        reactions=Sum("reactions"),
        kcat=Sum("kcat"),
        km=Sum("km"),
        kcat_km=Sum("kcat_km"),
        jobs=Count("pk"),
    )
    completed = Job.objects.filter(status="Completed").count()
    payload = {
        "schema_version": VERSION,
        "scope": "all_time",
        "generated_at": now.isoformat(),
        "jobs_completed": completed,
        "reactions_completed": totals["reactions"] or 0,
        "unique_protein_sequences": AboutJobProtein.objects.filter(
            summary__job__status="Completed", summary__version=VERSION
        ).aggregate(n=Count("sequence_sha256", distinct=True))["n"],
        "kcat_predictions_completed": totals["kcat"] or 0,
        "km_predictions_completed": totals["km"] or 0,
        "kcat_km_predictions_completed": totals["kcat_km"] or 0,
        "jobs_with_unavailable_results": completed - totals["jobs"],
    }
    payload["parameter_predictions_completed"] = sum(
        payload[k]
        for k in (
            "kcat_predictions_completed",
            "km_predictions_completed",
            "kcat_km_predictions_completed",
        )
    )
    row.payload = json.dumps(payload, separators=(",", ":"))
    row.generated_at, row.is_stale = now, False
    row.save(update_fields=["payload", "generated_at", "is_stale", "updated_at"])
    try:
        cache.delete(MEMORY_KEY)
    except Exception:
        log.warning("About statistics memory cache unavailable")
    return payload


def _deserialise_payload(raw):
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def _fresh(payload):
    try:
        generated = timezone.datetime.fromisoformat(payload["generated_at"])
        age = (timezone.now() - generated).total_seconds()
        return (
            payload.get("schema_version") == VERSION
            and 0 <= age < MAX_AGE
            and all(isinstance(payload[k], int) and payload[k] >= 0 for k in COUNTS)
        )
    except (KeyError, TypeError, ValueError):
        return False


def get_about_stats():
    """At most one small row lookup on cache miss; no file or aggregate queries."""
    try:
        payload = cache.get(MEMORY_KEY)
    except Exception:
        payload = None
    if _fresh(payload):
        return payload
    row = AboutStatsCache.objects.filter(key=CACHE_KEY).only("payload").first()
    payload = _deserialise_payload(row.payload) if row else None
    if not _fresh(payload):
        raise StatsNotReady("Usage statistics are being updated")
    try:
        cache.set(MEMORY_KEY, payload, timeout=60)
    except Exception:
        pass
    return payload
