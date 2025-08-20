from datetime import datetime, timedelta, timezone
from django_redis import get_redis_connection
from django.db import transaction

DAILY_LIMIT = 20_000

def get_client_ip(request) -> str:
    # Adjust if you sit behind a trusted proxy; otherwise REMOTE_ADDR is fine.
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "0.0.0.0")

def _seconds_until_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    reset = datetime.combine((now + timedelta(days=1)).date(), datetime.min.time(), tzinfo=timezone.utc)
    return int((reset - now).total_seconds())

def _key(ip: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"quota:{today}:{ip}"

def get_or_create_user(ip: str):
    """Get or create user record for IP tracking."""
    from api.models import ApiUser
    user, created = ApiUser.objects.get_or_create(
        ip_address=ip,
        defaults={'first_seen': datetime.now(timezone.utc)}
    )
    if not created:
        user.last_seen = datetime.now(timezone.utc)
        user.save(update_fields=['last_seen'])
    return user

def get_user_daily_limit(ip: str) -> int:
    """Get the daily limit for a specific IP address."""
    try:
        from api.models import ApiUser
        user = ApiUser.objects.get(ip_address=ip)
        if user.is_blocked:
            return 0
        return user.effective_daily_limit
    except ApiUser.DoesNotExist:
        return DAILY_LIMIT

def get_quota_usage(ip: str) -> dict:
    """Get current quota usage for an IP."""
    r = get_redis_connection("default")
    key = _key(ip)
    current_usage = r.get(key)
    current_usage = int(current_usage) if current_usage else 0
    
    daily_limit = get_user_daily_limit(ip)
    remaining = max(0, daily_limit - current_usage)
    ttl = _seconds_until_midnight_utc()
    
    return {
        'used': current_usage,
        'remaining': remaining,
        'limit': daily_limit,
        'reset_in_seconds': ttl
    }

def reserve_or_reject(ip: str, requested: int):
    """
    Atomically reserve `requested` units for today's quota for this IP.
    Returns (allowed: bool, remaining_after: int, seconds_to_reset: int).
    """
    # Update user record
    user = get_or_create_user(ip)
    
    # Check if user is blocked
    if user.is_blocked:
        return False, 0, _seconds_until_midnight_utc()
    
    r = get_redis_connection("default")
    key = _key(ip)
    ttl = _seconds_until_midnight_utc()
    daily_limit = user.effective_daily_limit

    lua = r.register_script("""
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
    """)
    allowed, remaining = lua(keys=[key], args=[daily_limit, requested, ttl])
    return bool(allowed), int(remaining), ttl

def credit_back(ip: str, amount: int):
    """Decrease today's counter for this IP by `amount` (not below zero)."""
    if not ip or amount <= 0:
        return
    r   = get_redis_connection("default")
    key = _key(ip)
    ttl = _seconds_until_midnight_utc()

    lua = r.register_script("""
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
    """)
    lua(keys=[key], args=[amount, ttl])
