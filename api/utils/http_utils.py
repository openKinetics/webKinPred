"""
HTTP utilities for request validation and response handling.
"""
from typing import Optional, Tuple
from django.http import JsonResponse, HttpRequest


def validate_post_request(request: HttpRequest) -> Optional[JsonResponse]:
    """
    Validate that the request uses POST method.
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        JsonResponse with error if invalid, None if valid
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed."}, status=405)
    return None


def validate_post_request_similarity(request: HttpRequest) -> Optional[JsonResponse]:
    """
    Validate that the request uses POST method (for similarity endpoints).
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        JsonResponse with error if invalid, None if valid
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests allowed"}, status=405)
    return None


def extract_file_from_request(request: HttpRequest) -> Tuple[Optional[object], Optional[JsonResponse]]:
    """
    Extract file from POST request.
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        Tuple of (file_object, error_response). If file exists, error_response is None.
        If file doesn't exist, file_object is None and error_response contains the error.
    """
    file = request.FILES.get("file")
    if not file:
        return None, JsonResponse({"error": "No file provided."}, status=400)
    return file, None


def extract_csv_file_from_request(request: HttpRequest) -> Tuple[Optional[object], Optional[JsonResponse]]:
    """
    Extract CSV file from POST request (for similarity analysis).
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        Tuple of (csv_file_object, error_response). If file exists, error_response is None.
        If file doesn't exist, file_object is None and error_response contains the error.
    """
    if "file" not in request.FILES:
        return None, JsonResponse({"error": "CSV file not provided"}, status=400)
    
    csv_file = request.FILES["file"]
    return csv_file, None


def extract_session_id_from_request(request: HttpRequest) -> Tuple[Optional[str], Optional[JsonResponse]]:
    """
    Extract session_id from POST request.
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        Tuple of (session_id, error_response). If session_id exists, error_response is None.
        If session_id doesn't exist, session_id is None and error_response contains the error.
    """
    session_id = request.POST.get("session_id")
    if not session_id:
        return None, JsonResponse({"error": "session_id required"}, status=400)
    return session_id, None


def extract_validation_session_id(request: HttpRequest) -> str:
    """
    Extract validation session ID from POST request, with default fallback.
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        Session ID string (defaults to "default" if not provided)
    """
    return request.POST.get("validationSessionId") or "default"


def validate_post_with_file_request(request: HttpRequest) -> Optional[JsonResponse]:
    """
    Validate that request is POST and contains a file.
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        JsonResponse with error if invalid, None if valid
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed."}, status=405)
    
    if "file" not in request.FILES:
        return None, JsonResponse({"error": "No file provided."}, status=400)
    
    return None


def extract_csv_file_with_validation(request: HttpRequest) -> Tuple[Optional[object], Optional[JsonResponse]]:
    """
    Extract and validate CSV file from POST request with format checking.
    
    Args:
        request: Django HttpRequest object
        
    Returns:
        Tuple of (csv_file_object, error_response). If file is valid, error_response is None.
        If file is invalid or missing, file_object is None and error_response contains the error.
    """
    if request.method != "POST":
        return None, JsonResponse({"error": "Only POST method is allowed."}, status=405)
    
    if "file" not in request.FILES:
        return None, JsonResponse({"error": "No file provided."}, status=400)
    
    file = request.FILES["file"]
    
    # Check file format
    if not file.name.endswith(".csv"):
        return None, JsonResponse(
            {"error": "File format not supported. Please upload a CSV file."},
            status=400,
        )
    
    return file, None
