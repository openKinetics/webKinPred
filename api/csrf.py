from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie

@ensure_csrf_cookie
def get_csrf(request):

    return JsonResponse({"detail": "CSRF cookie set"})
