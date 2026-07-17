import logging

import io
import json
import threading
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from api.services.progress_service import push_line, finish_session
from api.services.similarity_service import analyze_sequence_similarity
from api.utils.http_utils import (
    validate_post_request_similarity,
    extract_csv_file_from_request,
    extract_validation_session_id,
)

_log = logging.getLogger(__name__)


@csrf_exempt
def sequence_similarity_summary(request):
    """Analyze protein sequence similarity against target databases."""
    _log.info(
        "Sequence similarity summary requested",
        extra={"event": "similarity.summary_requested"},
    )

    # Validate request method
    method_error = validate_post_request_similarity(request)
    if method_error:
        return method_error

    # Extract session ID
    session_id = extract_validation_session_id(request)

    try:
        # Extract CSV file from request
        csv_file, file_error = extract_csv_file_from_request(request)
        if file_error:
            return file_error

        # Read into memory so background thread can access it even if request closes
        csv_bytes = csv_file.read()
        csv_file_mem = io.BytesIO(csv_bytes)

        def stream_response():
            result_container = []
            error_container = []

            def worker():
                try:
                    res = analyze_sequence_similarity(csv_file_mem, session_id=session_id)
                    result_container.append(res)
                except Exception as e:
                    error_container.append(e)

            push_line(session_id, "==> Starting MMseqs2 similarity analysis")
            t = threading.Thread(target=worker)
            t.start()

            # Yield spaces to keep the connection alive (prevents a 524 Cloudflare timeout)
            while t.is_alive():
                yield b" "
                t.join(timeout=10.0)

            # Execution finished
            try:
                if error_container:
                    e = error_container[0]
                    if isinstance(e, ValueError):
                        _log.warning(
                            "Sequence similarity validation failed",
                            extra={
                                "event": "similarity.validation_failed",
                                "session_id": session_id,
                                "exception_type": type(e).__name__,
                            },
                        )
                        push_line(session_id, f"[VALIDATION ERROR] {e}")
                        yield json.dumps({"error": str(e)}).encode("utf-8")
                    else:
                        _log.exception(
                            "Sequence similarity failed",
                            extra={
                                "event": "similarity.failed",
                                "session_id": session_id,
                                "exception_type": type(e).__name__,
                            },
                        )
                        push_line(session_id, f"[EXCEPTION] {e}")
                        yield json.dumps({"error": str(e)}).encode("utf-8")
                else:
                    push_line(session_id, "==> Similarity histograms computed successfully")
                    yield json.dumps(result_container[0]).encode("utf-8")
            finally:
                finish_session(session_id)

        response = StreamingHttpResponse(stream_response(), content_type="application/json")
        response['X-Accel-Buffering'] = 'no'
        return response

    except Exception as e:
        _log.exception(
            "Expected error setting up similarity background thread",
            extra={
                "event": "similarity.failed_setup",
                "session_id": session_id,
                "exception_type": type(e).__name__,
            },
        )
        push_line(session_id, f"[EXCEPTION] {e}")
        finish_session(session_id)
        return JsonResponse({"error": str(e)}, status=500)
