from django.http import JsonResponse

class ApiKeyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith('/api/'):
            origin = request.headers.get('Origin')
            if not origin:
                origin = request.headers.get('Host')
            print(f"Origin: {origin}")
            if origin not in ['kineticxpredictor.humanmetabolism.org', 'http://localhost:3000']:
                return JsonResponse({'error': 'Forbidden: Invalid Origin.'}, status=403)

        response = self.get_response(request)
        return response

