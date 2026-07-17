from datetime import datetime, timedelta, timezone
from django_redis import get_redis_connection

DAILY_LIMIT = 20_000
API_KEY_SUBJECT_PREFIX = "apikey"


def get_client_ip(request) -> str:
    # Adjust if you sit behind a trusted proxy; otherwise REMOTE_ADDR is fine.
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "0.0.0.0")


def _seconds_until_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    reset = datetime.combine(
        (now + timedelta(days=1)).date(), datetime.min.time(), tzinfo=timezone.utc
    )
    return int((reset - now).total_seconds())


def _key(subject: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"quota:{today}:{subject}"


def api_key_quota_subject(api_key_or_id) -> str:
    """
    Build a stable quota subject string for an API key.

    Accepts either an ApiKey instance or a numeric key ID.
    """
    key_id = getattr(api_key_or_id, "pk", api_key_or_id)
    return f"{API_KEY_SUBJECT_PREFIX}:{key_id}"


def _parse_api_key_subject(subject: str) -> int | None:
    prefix = f"{API_KEY_SUBJECT_PREFIX}:"
    if not isinstance(subject, str) or not subject.startswith(prefix):
        return None
    try:
        return int(subject[len(prefix) :])
    except (TypeError, ValueError):
        return None


def get_or_create_user(ip: str):
    """Get or create user record for IP tracking."""
    from api.models import ApiUser

    user, created = ApiUser.objects.get_or_create(
        ip_address=ip, defaults={"first_seen": datetime.now(timezone.utc)}
    )
    if not created:
        user.last_seen = datetime.now(timezone.utc)
        user.save(update_fields=["last_seen"])
    return user


def get_api_key_daily_limit(api_key) -> int:
    """
    Resolve the effective daily limit for an ApiKey.

    If the owning user is blocked, limit is 0 regardless of key settings.
    """
    user = api_key.user
    if user.is_blocked:
        return 0
    return max(user.effective_daily_limit, api_key.custom_daily_limit or 0)


def get_user_daily_limit(subject: str) -> int:
    """
    Get the daily limit for a quota subject (IP or API-key subject).
    """
    api_key_id = _parse_api_key_subject(subject)
    if api_key_id is not None:
        from api.models import ApiKey

        try:
            api_key = ApiKey.objects.select_related("user").get(pk=api_key_id)
        except ApiKey.DoesNotExist:
            return DAILY_LIMIT
        return get_api_key_daily_limit(api_key)

    try:
        from api.models import ApiUser

        user = ApiUser.objects.get(ip_address=subject)
        if user.is_blocked:
            return 0
        return user.effective_daily_limit
    except ApiUser.DoesNotExist:
        return DAILY_LIMIT


def get_quota_usage(subject: str, daily_limit: int | None = None) -> dict:
    """
    Get current quota usage for a quota subject.

    If ``daily_limit`` is not provided, the function falls back to the
    IP-based ApiUser lookup for backward compatibility.
    """
    r = get_redis_connection("default")
    key = _key(subject)
    current_usage = r.get(key)
    current_usage = int(current_usage) if current_usage else 0

    if daily_limit is None:
        daily_limit = get_user_daily_limit(subject)
    remaining = max(0, daily_limit - current_usage)
    ttl = _seconds_until_midnight_utc()

    return {
        "used": current_usage,
        "remaining": remaining,
        "limit": daily_limit,
        "reset_in_seconds": ttl,
    }


def reserve_or_reject(subject: str, requested: int, daily_limit: int | None = None):
    """
    Atomically reserve ``requested`` units for today's quota for ``subject``.

    If ``daily_limit`` is omitted, this function uses the legacy IP-based
    ApiUser flow (including block checks and per-user custom limits).

    If ``daily_limit`` is provided, caller is responsible for block checks.
    Returns (allowed: bool, remaining_after: int, seconds_to_reset: int).
    """
    if daily_limit is None:
        # Legacy IP mode keeps ApiUser records up to date.
        if _parse_api_key_subject(subject) is None:
            user = get_or_create_user(subject)
            if user.is_blocked:
                return False, 0, _seconds_until_midnight_utc()
            daily_limit = user.effective_daily_limit
        else:
            daily_limit = get_user_daily_limit(subject)
            if daily_limit <= 0:
                return False, 0, _seconds_until_midnight_utc()

    r = get_redis_connection("default")
    key = _key(subject)
    ttl = _seconds_until_midnight_utc()

    lua = r.register_script(
        """
    local key     = KEYS[1]
    local limit   = tonumber(ARGV[1])
    local req     = tonumber(ARGV[2])
    local ttl_sec = tonumber(ARGV[3])

    local cur = redis.call('GET', key)
    if not cur then
        redis.call('SET', key, 0, 'EX', ttl_sec)
        cur = 0
    else
        cur = tonumber(cur)
    end

    if (cur + req) > limit then
        return {0, limit - cur}
    else
        local new = redis.call('INCRBY', key, req)
        if redis.call('TTL', key) < 0 then redis.call('EXPIRE', key, ttl_sec) end
        return {1, limit - new}
    end
    """
    )
    allowed, remaining = lua(keys=[key], args=[daily_limit, requested, ttl])
    return bool(allowed), int(remaining), ttl


def credit_back(subject: str, amount: int):
    """Decrease today's counter for ``subject`` by ``amount`` (not below zero)."""
    if not subject or amount <= 0:
        return
    r = get_redis_connection("default")
    key = _key(subject)
    ttl = _seconds_until_midnight_utc()

    lua = r.register_script(
        """
    local key     = KEYS[1]
    local amount  = tonumber(ARGV[1])
    local ttl_sec = tonumber(ARGV[2])

    local cur = redis.call('GET', key)
    if not cur then
        return 0
    end
    cur = tonumber(cur)
    local new = cur - amount
    if new < 0 then new = 0 end
    redis.call('SET', key, new, 'EX', ttl_sec)
    return new
    """
    )
    lua(keys=[key], args=[amount, ttl])


def get_user_quota_subject(user) -> str:
    """
    Resolve the preferred quota subject for an ApiUser.

    If the user has an active API key, quota is tracked by API key ID.
    Otherwise, fallback to legacy IP-based tracking.
    """
    from api.models import ApiKey

    try:
        api_key = user.api_key
    except ApiKey.DoesNotExist:
        return user.ip_address
    if api_key.is_active:
        return api_key_quota_subject(api_key.pk)
    return user.ip_address


def get_user_quota_daily_limit(user) -> int:
    """
    Resolve the effective daily limit for a user's active quota subject.
    """
    if user.is_blocked:
        return 0

    from api.models import ApiKey

    try:
        api_key = user.api_key
    except ApiKey.DoesNotExist:
        return user.effective_daily_limit

    if api_key.is_active:
        return get_api_key_daily_limit(api_key)
    return user.effective_daily_limit


def get_all_user_quota_subjects(user) -> list[str]:
    """
    Return all possible quota subjects for a user (legacy + API key subject).

    Useful for admin tools that need to clear both old and new counters.
    """
    subjects = [user.ip_address]

    from api.models import ApiKey

    try:
        api_key = user.api_key
    except ApiKey.DoesNotExist:
        return subjects

    key_subject = api_key_quota_subject(api_key.pk)
    if key_subject not in subjects:
        subjects.append(key_subject)
    return subjects
