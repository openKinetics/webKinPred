# api/tasks.py

from celery import shared_task
from django.conf import settings
from .models import Job
import pandas as pd
import os
from django.utils import timezone
import subprocess
def run_prediction_subprocess(command, job):
    """
    Run a prediction subprocess and update job progress based on stdout.

    Parameters:
    - command: List of command-line arguments to run the subprocess.
    - job: Job object to update progress.
    """
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read stdout line by line
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            # Process the line
            print("Subprocess output:", line.strip())
            # Check if it's a progress update
            if line.startswith("Progress:"):
                # Extract the number of predictions made
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        progress = parts[1]
                        predictions_made, total_predictions = progress.split('/')
                        predictions_made = int(predictions_made)
                        total_predictions = int(total_predictions)
                        # Update the job object
                        job.predictions_made = predictions_made
                        job.total_predictions = total_predictions
                        job.save()
                    except Exception as e:
                        print("Error parsing progress update:", e)
            else:
                # Handle other output if needed
                pass

        # Wait for the subprocess to finish
        process.wait()

        if process.returncode != 0:
            # An error occurred
            raise subprocess.CalledProcessError(process.returncode, process.args)

    except Exception as e:
        print("An error occurred while running the subprocess:")
        print(e)
        raise e


@shared_task
def run_dlkcat_predictions(job_id):
    from .models import Job
    import os
    import pandas as pd
    from django.utils import timezone
    from django.conf import settings
    from api.dlkcat import dlkcat_predictions

    job = Job.objects.get(job_id=job_id)
    job_dir = os.path.join(settings.MEDIA_ROOT, 'jobs', str(job.job_id))
    input_file_path = os.path.join(job_dir, 'input.csv')

    # Update job status to 'Processing'
    job.status = 'Processing'
    job.save()

    try:
        # Read the CSV input file
        df = pd.read_csv(input_file_path)
        sequences = df['Protein Sequence'].tolist()
        substrates = df['Substrate'].tolist()
        protein_ids = df.get('Protein Accession Number', []).tolist()

        # Run the predictions (molecule processing and tracking are handled inside)
        predictions, invalid_indices = dlkcat_predictions(
            sequences=sequences,
            substrates=substrates,
            jobID=job.job_id,
            protein_ids=protein_ids
        )

        # Save results to output file
        output_file_path = os.path.join(job_dir, 'output.csv')
        df['kcat (1/s)'] = predictions
        # Reorder columns to put 'kcat (1/s)' first
        cols = ['kcat (1/s)'] + [col for col in df.columns if col != 'kcat (1/s)']
        df = df[cols]
        df.to_csv(output_file_path, index=False)

        # Update job with invalid indices information
        if invalid_indices:
            job.error_message = (
                f"Predictions could not be made for {len(invalid_indices)} row(s) due to invalid SMILES/InChI.\n"
                f"Indices:\n- " + "\n- ".join(map(str, invalid_indices))
            )
        else:
            job.error_message = ""

        # Update job status to 'Completed'
        job.status = 'Completed'
        job.completion_time = timezone.now()
        job.output_file.name = os.path.relpath(output_file_path, settings.MEDIA_ROOT)
        job.save()

    except Exception as e:
        # Update job status to 'Failed' and save error message
        job.status = 'Failed'
        job.error_message = str(e)
        job.completion_time = timezone.now()
        job.save()

@shared_task
def run_turnup_predictions(job_id):
    from .models import Job
    import os
    import pandas as pd
    from django.utils import timezone
    from django.conf import settings
    from api.turnup import turnup_predictions

    job = Job.objects.get(job_id=job_id)
    job_dir = os.path.join(settings.MEDIA_ROOT, 'jobs', str(job.job_id))
    input_file_path = os.path.join(job_dir, 'input.csv')

    # Update job status to 'Processing'
    job.status = 'Processing'
    job.save()

    try:
        # Read the CSV input file
        df = pd.read_csv(input_file_path)
        sequences = df['Protein Sequence'].tolist()
        substrates = df['Substrates'].tolist()
        products = df['Products'].tolist()
        protein_ids = df.get('Protein Accession Number', []).tolist()

        # Run the predictions (molecule processing and tracking are handled inside)
        predictions, invalid_indices = turnup_predictions(
            sequences=sequences,
            substrates=substrates,
            products=products,
            jobID=job.job_id,
            protein_ids=protein_ids
        )

        # Save results to output file
        output_file_path = os.path.join(job_dir, 'output.csv')
        df['kcat (1/s)'] = predictions
        # Reorder columns to put 'kcat (1/s)' first
        cols = ['kcat (1/s)'] + [col for col in df.columns if col != 'kcat (1/s)']
        df = df[cols]
        df.to_csv(output_file_path, index=False)

        # Update job with invalid indices information
        if invalid_indices:
            job.error_message = (
                f"Predictions could not be made for {len(invalid_indices)} row(s): \n"
                f"{', '.join(map(str, invalid_indices))} due to invalid SMILES/InChI."
            )
        else:
            job.error_message = ""

        # Update job status to 'Completed'
        job.status = 'Completed'
        job.completion_time = timezone.now()
        job.output_file.name = os.path.relpath(output_file_path, settings.MEDIA_ROOT)
        job.save()

    except Exception as e:
        # Update job status to 'Failed' and save error message
        job.status = 'Failed'
        job.error_message = str(e)
        job.completion_time = timezone.now()
        job.save()

@shared_task
def run_eitlem_predictions(job_id):
    from .models import Job
    import os
    import pandas as pd
    from django.utils import timezone
    from django.conf import settings
    from api.eitlem import eitlem_predictions
    job = Job.objects.get(job_id=job_id)
    job_dir = os.path.join(settings.MEDIA_ROOT, 'jobs', str(job.job_id))
    input_file_path = os.path.join(job_dir, 'input.csv')

    # Update job status to 'Processing'
    job.status = 'Processing'
    job.save()

    try:
        # Read the CSV input file
        df = pd.read_csv(input_file_path)
        sequences = df['Protein Sequence'].tolist()
        substrates = df['Substrate'].tolist()
        protein_ids = df.get('Protein Accession Number', []).tolist()

        # Run the predictions (molecule processing and tracking are handled inside)
        predictions, invalid_indices = eitlem_predictions(
            sequences=sequences,
            substrates=substrates,
            jobID=job.job_id,
            protein_ids=protein_ids,
            kinetics_type=job.prediction_type.upper()
        )

        # Save results to output file
        output_file_path = os.path.join(job_dir, 'output.csv')
        col_name = 'kcat (1/s)' if job.prediction_type.lower() == 'kcat' else 'KM (mM)'
        df[col_name] = predictions
        # Reorder columns to put the new column first
        cols = [col_name] + [col for col in df.columns if col != col_name]
        df = df[cols]
        df.to_csv(output_file_path, index=False)

        # Update job with invalid indices information
        if invalid_indices:
            job.error_message = (
                f"Predictions could not be made for {len(invalid_indices)} row(s) \n: {', '.join(map(str, invalid_indices))} "
                f"due to invalid SMILES/InChI."
            )
        else:
            job.error_message = ""

        # Update job status to 'Completed'
        job.status = 'Completed'
        job.completion_time = timezone.now()
        job.output_file.name = os.path.relpath(output_file_path, settings.MEDIA_ROOT)
        job.save()

    except Exception as e:
        # Update job status to 'Failed' and save error message
        job.status = 'Failed'
        job.error_message = str(e)
        job.completion_time = timezone.now()
        job.save()


# tasks.py
@shared_task
def run_both_predictions(job_id, kcat_method, km_method):
    from .models import Job
    import os
    import pandas as pd
    from django.utils import timezone
    from django.conf import settings
    from api.dlkcat import dlkcat_predictions
    from api.turnup import turnup_predictions
    from api.eitlem import eitlem_predictions

    job = Job.objects.get(job_id=job_id)
    job_dir = os.path.join(settings.MEDIA_ROOT, 'jobs', str(job.job_id))
    input_file_path = os.path.join(job_dir, 'input.csv')

    # Update job status to 'Processing'
    job.status = 'Processing'
    job.predictions_made = 0
    job.total_predictions = 0
    job.save()

    try:
        # Read the CSV input file
        df = pd.read_csv(input_file_path)
        sequences = df['Protein Sequence'].tolist()
        protein_ids = df.get('Protein Accession Number', []).tolist()

        # Initialize the DataFrame to store results
        results_df = df.copy()
        invalid_indices = set()

        # Run kcat predictions
        if kcat_method == 'DLKcat':
            if 'Substrate' not in df.columns:
                raise ValueError('Missing "Substrate" column required for DLKcat kcat predictions.')
            substrates = df['Substrate'].tolist()

            # Run DLKcat predictions
            kcat_predictions, kcat_invalid_indices = dlkcat_predictions(
                sequences=sequences,
                substrates=substrates,
                jobID=job.job_id,
                protein_ids=protein_ids
            )
            results_df['kcat (1/s)'] = kcat_predictions
            invalid_indices.update(kcat_invalid_indices)

        elif kcat_method == 'EITLEM':
            if 'Substrate' not in df.columns:
                raise ValueError('Missing "Substrate" column required for EITLEM kcat predictions.')
            substrates = df['Substrate'].tolist()

            # Run EITLEM kcat predictions
            kcat_predictions, kcat_invalid_indices = eitlem_predictions(
                sequences=sequences,
                substrates=substrates,
                jobID=job.job_id,
                protein_ids=protein_ids,
                kinetics_type='KCAT'
            )
            results_df['kcat (1/s)'] = kcat_predictions
            invalid_indices.update(kcat_invalid_indices)

        elif kcat_method == 'TurNup':
            if 'Substrates' not in df.columns or 'Products' not in df.columns:
                raise ValueError('Missing "Substrates" or "Products" columns required for TurNup kcat predictions.')
            substrates = df['Substrates'].tolist()
            products = df['Products'].tolist()

            # Run TurNup predictions
            kcat_predictions, kcat_invalid_indices = turnup_predictions(
                sequences=sequences,
                substrates=substrates,
                products=products,
                jobID=job.job_id,
                protein_ids=protein_ids
            )
            results_df['kcat (1/s)'] = kcat_predictions
            invalid_indices.update(kcat_invalid_indices)
        else:
            raise ValueError('Invalid kcat method.')
        
        job.kcat_complete = True  # Mark that kcat predictions are complete
        job.save()
        # Reset predictions made for KM predictions
        job.predictions_made = 0
        job.total_predictions = 0
        job.save()

        # Run KM predictions
        if km_method == 'EITLEM':
            if 'Substrate' not in df.columns:
                raise ValueError('Missing "Substrate" column required for EITLEM KM predictions.')
            substrates = df['Substrate'].tolist()

            # Run EITLEM KM predictions
            km_predictions, km_invalid_indices = eitlem_predictions(
                sequences=sequences,
                substrates=substrates,
                jobID=job.job_id,
                protein_ids=protein_ids,
                kinetics_type='KM'
            )
            results_df['KM (mM)'] = km_predictions
            invalid_indices.update(km_invalid_indices)
        else:
            raise ValueError('Invalid KM method.')

        # Reorder columns to have 'kcat' and 'KM' at the front
        cols = ['kcat (1/s)', 'KM (mM)'] + [col for col in results_df.columns if col not in ['kcat (1/s)', 'KM (mM)']]
        results_df = results_df[cols]

        # Save results to output file
        output_file_path = os.path.join(job_dir, 'output.csv')
        results_df.to_csv(output_file_path, index=False)

        # Update job with invalid indices information
        if invalid_indices:
            invalid_indices = sorted(list(invalid_indices))
            job.error_message = (
                f"Predictions could not be made for {len(invalid_indices)} row(s): \n"
                f"{', '.join(map(str, invalid_indices))} due to invalid SMILES/InChI."
            )
        else:
            job.error_message = ""

        # Update job status to 'Completed'
        job.status = 'Completed'
        job.completion_time = timezone.now()
        job.output_file.name = os.path.relpath(output_file_path, settings.MEDIA_ROOT)
        job.save()

    except Exception as e:
        # Update job status to 'Failed' and save error message
        job.status = 'Failed'
        job.error_message = str(e)
        job.completion_time = timezone.now()
        job.save()
