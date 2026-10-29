from pathlib import Path

from django.conf import settings
from django.http import JsonResponse


def maintenance_active():
    return Path(settings.EXECUTION_MAINTENANCE_FILE).exists()


class ExecutionMaintenanceMiddleware:
    """Stop expensive work at ingress, before parsing and quota reservation."""

    blocked = {
        "/api/submit-job/",
        "/api/validate-input/",
        "/api/sequence-similarity-summary/",
        "/api/v1/submit/",
        "/api/v1/validate/",
    }

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        mutation = request.method not in {"GET", "HEAD", "OPTIONS"} and not (
            request.path.startswith("/internal/v2/") or request.path == "/api/cancel-validation/"
        )
        if maintenance_active() and (
            request.path in self.blocked or mutation or request.path.startswith("/admin/")
        ):
            response = JsonResponse(
                {"error": "Job execution is paused for maintenance.", "execution_state": "paused"},
                status=503,
            )
            response["Retry-After"] = "300"
            return response
        return self.get_response(request)
