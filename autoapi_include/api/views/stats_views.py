from django.http import JsonResponse
from django.views.decorators.http import require_GET

from api.services.about_stats_service import get_about_stats


@require_GET
def about_stats(request):
    return JsonResponse(get_about_stats())
