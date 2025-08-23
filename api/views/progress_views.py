
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse

from api.services.progress_service import sse_generator
@csrf_exempt
def progress_stream(request):
    """
    SSE endpoint: /api/progress-stream/?session_id=XYZ
    Streams logs for the given session_id.
    """
    session_id = request.GET.get("session_id")
    if not session_id:
        return JsonResponse({"error": "session_id query param required"}, status=400)
    # The start_session(session_id) call is REMOVED. It is no longer needed.
    response = StreamingHttpResponse(
        streaming_content=sse_generator(session_id),
        content_type="text/event-stream",
    )
    # Helpful SSE headers
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
