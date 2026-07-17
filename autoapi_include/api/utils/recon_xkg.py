"""
Authorization + parsing helpers for the undocumented ``recon_xkg`` submit flag.

``recon_xkg`` lets an *allowlisted* API key serve predictions from the
persistent memoization store instead of recomputing everything. It is
deliberately kept out of the public docs/OpenAPI schema. Unauthorized use is
silently ignored (treated as a normal job) rather than rejected with a
distinctive error, so the parameter's existence is not confirmed to callers —
attempts are only recorded server-side for audit.
"""

from __future__ import annotations

import logging
import time

_log = logging.getLogger(__name__)

# Short in-process allowlist cache so a hot submit path does not hit the DB on
# every request. Entries expire quickly so admin allow/deny changes take effect
# without a process restart.
_ALLOW_CACHE_TTL_SECONDS = 30.0
_allow_cache: dict[int, tuple[float, bool]] = {}


def coerce_recon_xkg(raw) -> bool:
    """Parse the multipart/JSON ``recon_xkg`` value into a bool (default False)."""
    from api.utils.job_utils import coerce_bool_param

    return coerce_bool_param(raw, default=False)


def is_recon_xkg_allowed(api_key) -> bool:
    """
    Return True if ``api_key`` is on the ReconXKG allowlist.

    Membership is the only authorization gate. Looks up an active
    :class:`ReconXkgAllowedKey` row, with a brief per-process TTL cache. Any
    lookup error fails closed (returns False).
    """
    if api_key is None or getattr(api_key, "pk", None) is None:
        return False

    now = time.monotonic()
    cached = _allow_cache.get(api_key.pk)
    if cached is not None and (now - cached[0]) < _ALLOW_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        from api.models import ReconXkgAllowedKey

        allowed = ReconXkgAllowedKey.objects.filter(
            api_key_id=api_key.pk, is_active=True
        ).exists()
    except Exception:
        _log.warning(
            "ReconXKG allowlist lookup failed; denying",
            extra={"event": "recon_xkg.allowlist_lookup_failed"},
            exc_info=True,
        )
        allowed = False

    _allow_cache[api_key.pk] = (now, allowed)
    return allowed


def resolve_recon_xkg(api_key, requested: bool) -> bool:
    """
    Resolve the effective recon_xkg flag for a submission and audit-log it.

    Returns True only when the flag was requested *and* the key is allowlisted.
    Unauthorized requests are logged (without leaking to the client) and
    downgraded to False.
    """
    if not requested:
        return False

    if is_recon_xkg_allowed(api_key):
        _log.info(
            "ReconXKG activated for submission",
            extra={
                "event": "recon_xkg.activated",
                "api_key_id": getattr(api_key, "pk", None),
            },
        )
        return True

    _log.warning(
        "ReconXKG requested by non-allowlisted key; ignoring (normal job)",
        extra={
            "event": "recon_xkg.unauthorized_attempt",
            "api_key_id": getattr(api_key, "pk", None),
        },
    )
    return False


def log_recon_xkg_outcome(
    *,
    api_key_id,
    public_id: str,
    row_count: int,
    hit_count: int,
    miss_count: int,
) -> None:
    """Audit-log the hit/miss outcome of an authorized ReconXKG job."""
    _log.info(
        "ReconXKG job cache outcome",
        extra={
            "event": "recon_xkg.cache_outcome",
            "api_key_id": api_key_id,
            "job_public_id": public_id,
            "row_count": row_count,
            "hit_count": hit_count,
            "miss_count": miss_count,
        },
    )
