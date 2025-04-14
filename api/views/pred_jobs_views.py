import pandas as pd
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from django.shortcuts import get_object_or_404
from ..models import Job
from ..tasks import run_dlkcat_predictions, run_turnup_predictions, run_eitlem_predictions, run_both_predictions

@csrf_exempt
def submit_job(request):
    if request.method == 'POST' and 'file' in request.FILES:
        file = request.FILES['file']
        prediction_type = request.POST.get('predictionType')
        kcat_method = request.POST.get('kcatMethod')
        km_method = request.POST.get('kmMethod')

        # Check if the file is a CSV
        if not file.name.endswith('.csv'):
            return JsonResponse({'error': 'File format not supported. Please upload a CSV file.'}, status=400)
    
        try:
            # Read the uploaded CSV file directly from the file object
            df = pd.read_csv(file)
        except Exception as e:
            return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)
        required_columns = ['Protein Sequence', 'Protein Accession Number']

        # Determine additional required columns based on the selected method
        if kcat_method == 'TurNup':
            required_columns.extend(['Substrates', 'Products'])
        elif kcat_method in ['DLKcat', 'EITLEM']:
            required_columns.append('Substrate')

        # Check if the required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return JsonResponse({'error': f'Missing required columns: {", ".join(missing_columns)}'}, status=400)
        
        # Create a Job object
        job = Job(
            prediction_type=prediction_type,
            kcat_method=kcat_method,
            km_method=km_method,
            status='Pending'
        )
        job.save()

        # Save the file to a directory associated with the job
        job_dir = os.path.join(settings.MEDIA_ROOT, 'jobs', str(job.job_id))
        os.makedirs(job_dir, exist_ok=True)
        file_path = os.path.join(job_dir, 'input.csv')

        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        if prediction_type == 'both':
            run_both_predictions.delay(job.job_id, kcat_method, km_method)
        elif prediction_type == 'kcat':
            pred_func = run_dlkcat_predictions if kcat_method == 'DLKcat' else run_turnup_predictions if kcat_method == 'TurNup' else run_eitlem_predictions    
            pred_func.delay(job.job_id)
        elif prediction_type == 'Km':
            pred_func = run_eitlem_predictions if km_method == 'EITLEM' else None
            if pred_func:
                pred_func.delay(job.job_id)
            else:
                return JsonResponse({'error': 'Invalid prediction type'}, status=400)
        else:
            return JsonResponse({'error': 'Invalid prediction type'}, status=400)


        return JsonResponse({
            'message': 'Job submitted successfully',
            'job_id': job.job_id
        })
    else:
        return JsonResponse({'error': 'File upload failed'}, status=400)

# views.py
def job_status(request, job_id):
    job = get_object_or_404(Job, job_id=job_id)
    response_data = {
        'job_id': job.job_id,
        'status': job.status,
        'submission_time': job.submission_time,
        'completion_time': job.completion_time,
        'error_message': job.error_message,
        'total_molecules': job.total_molecules,
        'molecules_processed': job.molecules_processed,
        'invalid_molecules': job.invalid_molecules,
        'total_predictions': job.total_predictions,
        'predictions_made': job.predictions_made,
    }
    if job.status == 'Completed' and job.output_file:
        response_data['output_file_url'] = settings.MEDIA_URL + job.output_file.name
    return JsonResponse(response_data)
