
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404

from ..models import Job
from api.services.job_service import process_job_submission, get_job_status_data
from api.utils.http_utils import extract_csv_file_with_validation


@csrf_exempt
def submit_job(request):
    """Submit a new prediction job."""
    # Extract and validate CSV file from request
    csv_file, file_error = extract_csv_file_with_validation(request)
    if file_error:
        return file_error
    
    # Process job submission
    error_response, success_data = process_job_submission(request, csv_file)
    
    if error_response:
        return error_response
    
    return JsonResponse(success_data)

def job_status(request, public_id):
    """Get status information for a specific job."""
    job = get_object_or_404(Job, public_id=public_id)
    response_data = get_job_status_data(job)
    return JsonResponse(response_data)