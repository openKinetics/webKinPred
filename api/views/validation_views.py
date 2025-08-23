from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from api.services.progress_service import cancel_session
from api.services.validation_service import validate_input_file
from api.utils.http_utils import (
    validate_post_request,
    extract_file_from_request,
    extract_session_id_from_request,
)

@csrf_exempt
def cancel_validation(request):
    """Cancel a validation session."""
    # Validate request method
    method_error = validate_post_request(request)
    if method_error:
        return method_error
    
    # Extract session ID from request
    session_id, extraction_error = extract_session_id_from_request(request)
    if extraction_error:
        return extraction_error
    
    # Cancel the session
    success = cancel_session(session_id)
    return JsonResponse({"ok": bool(success)})


@csrf_exempt
def validate_input(request):
    """Validate uploaded CSV file for substrate and protein data."""
    # Validate request method
    method_error = validate_post_request(request)
    if method_error:
        return method_error

    # Extract file from request
    csv_file, file_error = extract_file_from_request(request)
    if file_error:
        return file_error

    # Validate the file content
    validation_result = validate_input_file(csv_file)
    
    # Handle validation errors
    if "error" in validation_result:
        return JsonResponse(
            {"error": validation_result["error"]}, 
            status=validation_result["status_code"]
        )
    
    # Return successful validation result
    return JsonResponse({
        "invalid_substrates": validation_result["invalid_substrates"],
        "invalid_proteins": validation_result["invalid_proteins"],
        "length_violations": validation_result["length_violations"],
    }, status=validation_result["status_code"])