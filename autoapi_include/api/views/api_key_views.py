"""
Web-facing API key management views.

These endpoints are used by the React frontend to let visitors generate,
view, and revoke API keys — all keyed by the visitor's IP address.

Endpoints (mounted under /api/ in api/urls.py):
    GET   /api/api-key/          — check if the current IP already has a key
    POST  /api/api-key/generate/ — generate a new key for the current IP
    POST  /api/api-key/revoke/   — revoke the current IP's key
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from api.models import ApiKey, generate_api_key
from api.utils.quotas import get_client_ip, get_or_create_user


@csrf_exempt
@require_GET
def api_key_status(request):
    """
    Return whether the visitor's IP already has an active API key.

    Response:
        { "hasKey": true,  "keySuffix": "…a1b2c3" }
        { "hasKey": false }
    """
    ip = get_client_ip(request)

    try:
        api_key = ApiKey.objects.select_related("user").get(user__ip_address=ip, is_active=True)
        return JsonResponse(
            {
                "hasKey": True,
                "keySuffix": api_key.key[-6:],
            }
        )
    except ApiKey.DoesNotExist:
        return JsonResponse({"hasKey": False})


@csrf_exempt
@require_POST
def api_key_generate(request):
    """
    Generate a new API key for the visitor's IP address.

    If an active key already exists, return 409 Conflict.  The caller
    should revoke first and then generate.

    The full key is returned exactly once — subsequent calls to the
    status endpoint will only show the last 6 characters.
    """
    ip = get_client_ip(request)
    user = get_or_create_user(ip)

    if user.is_blocked:
        return JsonResponse(
            {"error": "Your IP address has been blocked. Contact the administrators."},
            status=403,
        )

    # Enforce one active key per IP.
    if ApiKey.objects.filter(user=user, is_active=True).exists():
        return JsonResponse(
            {"error": "An active API key already exists for your IP. Revoke it first."},
            status=409,
        )

    # OneToOneField means only one row per user can ever exist. If a revoked key
    # exists, reactivate it with a fresh token rather than trying to create a new row.
    try:
        api_key = ApiKey.objects.get(user=user)
        api_key.key = generate_api_key()
        api_key.is_active = True
        api_key.label = "Self-service (web)"
        api_key.save(update_fields=["key", "is_active", "label"])
    except ApiKey.DoesNotExist:
        api_key = ApiKey.objects.create(user=user, label="Self-service (web)")

    return JsonResponse(
        {
            "key": api_key.key,
            "keySuffix": api_key.key[-6:],
            "message": "Store this key securely — it will not be shown again.",
        },
        status=201,
    )


@csrf_exempt
@require_POST
def api_key_revoke(request):
    """
    Revoke (deactivate) the active API key for the visitor's IP.
    """
    ip = get_client_ip(request)

    updated = ApiKey.objects.filter(user__ip_address=ip, is_active=True).update(is_active=False)

    if updated:
        return JsonResponse({"revoked": True})
    else:
        return JsonResponse(
            {"error": "No active API key found for your IP."},
            status=404,
        )
