import pandas as pd
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from django.shortcuts import get_object_or_404
from ..models import Job
from ..tasks import run_dlkcat_predictions, run_turnup_predictions, run_eitlem_predictions, run_unikp_predictions, run_both_predictions

@csrf_exempt
def submit_job(request):
    if request.method == 'POST' and 'file' in request.FILES:
        print("Received POST to submit-job")
        file = request.FILES['file']
        prediction_type = request.POST.get('predictionType')
        kcat_method = request.POST.get('kcatMethod')
        km_method = request.POST.get('kmMethod')
        handleLongSeq = request.POST.get('handleLongSequences')
        assert handleLongSeq in ['truncate', 'skip'], "Invalid handleLongSeq value. Expected 'truncate' or 'skip'."
        # Check if the file is a CSV
        if not file.name.endswith('.csv'):
            return JsonResponse({'error': 'File format not supported. Please upload a CSV file.'}, status=400)
        try:
            # Read the uploaded CSV file directly from the file object
            df = pd.read_csv(file)
        except Exception as e:
            return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)
        required_columns = ['Protein Sequence']

        # Determine additional required columns based on the selected method
        if kcat_method == 'TurNup':
            required_columns.extend(['Substrates', 'Products'])
        elif kcat_method in ['DLKcat', 'EITLEM', 'UniKP']:
            required_columns.append('Substrate')
        else:
            return JsonResponse({'error': 'Invalid kcat method'}, status=400)
        # Check if the required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return JsonResponse({'error': f'Missing required columns: {", ".join(missing_columns)}'}, status=400)
        
        # Create a Job object
        job = Job(
            prediction_type=prediction_type,
            kcat_method=kcat_method,
            km_method=km_method,
            status='Pending',
            handle_long_sequences=handleLongSeq,
        )
        job.save()
        print("Saved Job:", job.job_id)
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
            method_to_func = {
                'DLKcat': run_dlkcat_predictions,
                'TurNup': run_turnup_predictions,
                'EITLEM': run_eitlem_predictions,
                'UniKP': run_unikp_predictions
            }
            pred_func = method_to_func.get(kcat_method)  
            from celery.result import AsyncResult
            job_id_int = int(job.job_id)
            task = pred_func.apply_async(args=[job_id_int])
            print("✅ Task ID:", task.id)

            # Just for debugging — check its status right after dispatch
            from celery.result import AsyncResult
            result = AsyncResult(task.id)
            print("🚦 Task state:", result.status)

            print("Dispatching task to Celery:", prediction_type, kcat_method, km_method)

        elif prediction_type == 'Km':
            method_to_func = {
                'EITLEM': run_eitlem_predictions,
                'UniKP': run_unikp_predictions
            }
            pred_func = method_to_func.get(km_method)
            if pred_func:
                pred_func.delay(job.job_id)
                print("Dispatching task to Celery:", prediction_type, kcat_method, km_method)
             
            else:
                return JsonResponse({'error': 'Invalid prediction type'}, status=400)
        else:
            return JsonResponse({'error': 'Invalid prediction type'}, status=400)

        print("Task dispatched")

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

@csrf_exempt
def detect_csv_format(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({'error': 'POST with CSV file required.'}, status=400)

    file = request.FILES['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return JsonResponse({'status': 'invalid', 'errors': [f'Error reading CSV: {str(e)}']}, status=400)

    errors = []
    required_cols = {'Protein Sequence'}
    has_substrate = 'Substrate' in df.columns
    has_substrates_products = 'Substrates' in df.columns and 'Products' in df.columns

    if not required_cols.issubset(df.columns):
        errors.append("Missing required column: 'Protein Sequence'")

    if not has_substrate and not has_substrates_products:
        errors.append("Missing substrate information: expected either 'Substrate' or both 'Substrates' and 'Products'")

    if errors:
        return JsonResponse({'status': 'invalid', 'errors': errors})

    if has_substrates_products:
        return JsonResponse({'status': 'valid', 'csv_type': 'multi'})
    elif has_substrate:
        return JsonResponse({'status': 'valid', 'csv_type': 'single'})
    else:
        return JsonResponse({'status': 'invalid', 'errors': ['Could not determine format.']}, status=400)
