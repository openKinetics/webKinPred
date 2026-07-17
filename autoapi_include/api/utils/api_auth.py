"""
API key authentication for the public REST API.

All v1 endpoints that require authentication should be decorated with
@require_api_key.  The decorator:

  1. Reads the "Authorization: Bearer <key>" header.
  2. Looks up the key in the database.
  3. Checks that the key is active and that the owner is not blocked.
  4. Stamps the key's last_used timestamp.
  5. Attaches request.api_user/request.api_key plus quota context fields.

The decorator also applies @csrf_exempt so that programmatic clients do not
need to obtain a CSRF cookie.
"""

from functools import wraps

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt


def require_api_key(view_func):
    """
    Decorator that authenticates requests via a Bearer token.

    Usage:

        @require_api_key
        def my_view(request):
            user = request.api_user
            key = request.api_key
            quota_subject = request.api_quota_subject
            ...

    On success, the following attributes are attached:
      - request.api_user          — the ApiUser record tied to this key
      - request.api_key           — the ApiKey instance used for auth
      - request.api_quota_subject — Redis quota subject tied to this API key
      - request.api_daily_limit   — effective daily limit for this key
      - request.api_request_ip    — source IP of the current request
      - request.api_ip            — legacy alias for the owner's registered IP
    """

    @csrf_exempt
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        from api.models import ApiKey  # local import to avoid circular imports

        auth_header = request.META.get("HTTP_AUTHORIZATION", "")

        if not auth_header.startswith("Bearer "):
            return JsonResponse(
                {
                    "error": (
                        "Authentication required. "
                        "Include the header: Authorization: Bearer <your_api_key>"
                    )
                },
                status=401,
            )

        token = auth_header[len("Bearer ") :].strip()

        try:
            api_key = ApiKey.objects.select_related("user").get(key=token, is_active=True)
        except ApiKey.DoesNotExist:
            return JsonResponse(
                {"error": "Invalid or revoked API key."},
                status=401,
            )

        if api_key.user.is_blocked:
            return JsonResponse(
                {"error": "This account has been suspended. Contact the administrators."},
                status=403,
            )

        # Record the most recent use of this key (non-blocking — best effort).
        api_key.last_used = timezone.now()
        api_key.save(update_fields=["last_used"])

        # Attach authenticated context for downstream handlers.
        from api.utils.quotas import (
            api_key_quota_subject,
            get_api_key_daily_limit,
            get_client_ip,
        )

        request.api_key = api_key
        request.api_user = api_key.user
        request.api_quota_subject = api_key_quota_subject(api_key.pk)
        request.api_daily_limit = get_api_key_daily_limit(api_key)
        request.api_request_ip = get_client_ip(request)
        request.api_ip = api_key.user.ip_address  # legacy compatibility

        return view_func(request, *args, **kwargs)

    return wrapper
