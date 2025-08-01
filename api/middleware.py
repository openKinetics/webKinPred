from django.http import JsonResponse
from django.conf import settings

class ApiKeyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Get the real client IP address from the X-Forwarded-For header, if available
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            # The client IP is the first in the list of forwarded IPs
            client_ip = x_forwarded_for.split(',')[0]
        else:
            # Fallback to REMOTE_ADDR if X-Forwarded-For is not available
            client_ip = request.META.get('REMOTE_ADDR', None)

        print(f"Client IP: {client_ip}")

        if request.path.startswith('/api/'):
            # Define allowed frontend IPs (for example, you might allow certain IPs or ranges)
            allowed_ips = settings.ALLOWED_FRONTEND_IPS
            
            if client_ip not in allowed_ips:
                return JsonResponse({'error': 'Forbidden: Unauthorized IP.'}, status=403)

        response = self.get_response(request)
        return response
