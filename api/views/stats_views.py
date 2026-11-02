import hashlib
import json

from api.services.about_stats_service import StatsNotReady, get_about_stats
from django.http import HttpResponseNotModified, JsonResponse
from django.views.decorators.http import require_GET


@require_GET
def about_stats(request):
    try:
        payload = get_about_stats()
    except StatsNotReady:
        response = JsonResponse({"detail": "Usage statistics are being updated"}, status=503)
        response["Cache-Control"] = "no-store"
        response["Retry-After"] = "60"
        return response
    etag = '"' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() + '"'
    response = (
        HttpResponseNotModified()
        if request.headers.get("If-None-Match") == etag
        else JsonResponse(payload)
    )
    response["ETag"] = etag
    # Revalidate snapshots at most a minute apart; never cache a snapshot past
    # its original freshness deadline, even if a worker is temporarily offline.
    from api.services.about_stats_service import MAX_AGE
    from django.utils import timezone

    age = max(
        0,
        (timezone.now() - timezone.datetime.fromisoformat(payload["generated_at"])).total_seconds(),
    )
    response["Cache-Control"] = (
        f"public, max-age={min(60, max(0, int(MAX_AGE - age)))}, must-revalidate"
    )
    return response
