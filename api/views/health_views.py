from api.execution.middleware import maintenance_active
from django.http import JsonResponse


def health_check(request):
    return JsonResponse(
        {
            "status": "healthy",
            "submissions_paused": maintenance_active(),
        }
    )
