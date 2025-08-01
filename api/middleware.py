from django.http import JsonResponse
from webKinPred import config_local
from django.conf import settings

class ApiKeyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Define allowed frontend IPs (e.g., 192.168.1.0/24 or the server IP where frontend is hosted)
        allowed_ips = settings.ALLOWED_FRONTEND_IPS
        client_ip = request.META.get('REMOTE_ADDR')
        if request.path.startswith('/api/'):
            # Check if the request is coming from an allowed IP
            if client_ip not in allowed_ips:
                return JsonResponse({'error': 'Forbidden: Unauthorized IP.'}, status=403)
        response = self.get_response(request)
        return response