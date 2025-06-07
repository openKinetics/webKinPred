# api/tasks.py
from celery import shared_task
from django.conf import settings
import pandas as pd
import os
from django.utils import timezone
import subprocess

from webKinPred.config_local import MODEL_LIMITS, SERVER_LIMIT
from api.models import Job
from api.utils.handle_long import get_valid_indices, truncate_sequences

from api.dlkcat import dlkcat_predictions
from api.turnup import turnup_predictions
from api.eitlem import eitlem_predictions
from api.unikp import unikp_predictions

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

# api/utils/predict_runner.py
import os, pandas as pd
from django.utils import timezone
from django.conf import settings

from api.utils.handle_long import get_valid_indices, truncate_sequences
from webKinPred.config_local import MODEL_LIMITS, SERVER_LIMIT   # central limits


def run_model(
    *,
    job,
    model_key: str,
    df: pd.DataFrame,
    pred_func,                       # callable -> (preds, invalid_idx)
    requires_cols: list[str],
    extra_kwargs: dict | None = None,
    output_col: str,
    handle_long: str = "skip",
):
    """
    Generic prediction runner.

    Parameters
    ----------
    job             : Job   – Django Job row (already fetched)
    model_key       : str   – key inside MODEL_LIMITS (e.g. 'dlkcat')
    df              : DataFrame of the CSV
    pred_func       : callable(**kwargs) -> (preds, invalid_idxs)
    requires_cols   : list  – columns that must exist in df
    extra_kwargs    : dict  – pass-through to pred_func (e.g. kinetics_type)
    output_col      : str   – column to write predictions into
    handle_long     : 'skip'|'truncate'
    """

    extra_kwargs = extra_kwargs or {}

    # ------- 1. validate required columns ----------------------------------
    missing = [c for c in requires_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) required for {model_key}: {', '.join(missing)}")

    # ------- 2. gather core inputs ----------------------------------------
    sequences = df["Protein Sequence"].tolist()
    limit = min(SERVER_LIMIT, MODEL_LIMITS[model_key])

    # ------- 3. length handling -------------------------------------------
    if handle_long == "truncate":
        sequences_proc, valid_idx = truncate_sequences(sequences, limit)
    else:   # skip
        valid_idx = get_valid_indices(sequences, limit, mode="skip")
        sequences_proc = [sequences[i] for i in valid_idx]

    # ------- 4. model-specific other inputs --------------------------------
    kwargs = {"sequences": sequences_proc, "public_id": job.public_id} | extra_kwargs

    if model_key == "DLKcat":
        kwargs["substrates"] = [df["Substrate"][i] for i in valid_idx]
    elif model_key == "TurNup":
        kwargs["substrates"] = [df["Substrates"][i] for i in valid_idx]
        kwargs["products"]   = [df["Products"][i]   for i in valid_idx]
    elif model_key in {"EITLEM", "UniKP"}:
        kwargs["substrates"] = [df["Substrate"][i] for i in valid_idx]
    # (extend for more models as needed)

    # ------- 5. run prediction --------------------------------------------
    full_preds = ["" for _ in sequences]
    invalid_global: list[int] = []

    if valid_idx:                      # only call model if anything to predict
        pred_subset, invalid_subset = pred_func(**kwargs)

        for i, p in zip(valid_idx, pred_subset):
            full_preds[i] = p

        invalid_global = [valid_idx[i] for i in invalid_subset]
    # ------- 6. write CSV --------------------------------------------------
    df.insert(0, output_col, full_preds)
    csv_out = os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "output.csv")
    df.to_csv(csv_out, index=False)


    # ------- 7. update job -------------------------------------------------
    job.output_file.name = os.path.relpath(csv_out, settings.MEDIA_ROOT)
    job.error_message = (
        f"Predictions could not be made for {len(invalid_global)} row(s): "
        + ", ".join(map(str, invalid_global))
    ) if invalid_global else ""
    return csv_out
# ------------------------------------------------------------ DLKcat
@shared_task
def run_dlkcat_predictions(public_id):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()

    try:
        df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv"))

        run_model(
            job            = job,
            model_key      = "DLKcat",
            df             = df,
            pred_func      = dlkcat_predictions,
            requires_cols  = ["Substrate"],
            output_col     = "kcat (1/s)",
            handle_long    = job.handle_long_sequences,
        )

        job.status = "Completed"
        job.completion_time = timezone.now()
        job.save()

    except Exception as e:
        job.status = "Failed"
        job.error_message = str(e)
        job.completion_time = timezone.now()
        job.save()

# ------------------------------------------------------------ TurNup
@shared_task
def run_turnup_predictions(public_id):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()

    try:
        df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv"))

        run_model(
            job           = job,
            model_key     = "TurNup",
            df            = df,
            pred_func     = turnup_predictions,
            requires_cols = ["Substrates", "Products"],
            output_col    = "kcat (1/s)",
            handle_long   = job.handle_long_sequences,
            needs_multi   = True,
        )
        job.status = "Completed"; job.completion_time = timezone.now(); job.save()

    except Exception as e:
        job.status = "Failed"; job.error_message = str(e); job.completion_time = timezone.now(); job.save()

# ------------------------------------------------------------ EITLEM
@shared_task
def run_eitlem_predictions(public_id):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()

    try:
        df = pd.read_csv(
            os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv")
        )

        kin_flag   = job.prediction_type.upper()          # “KCAT” | “KM”
        out_col    = "kcat (1/s)" if kin_flag == "KCAT" else "KM (mM)"

        run_model(
            job           = job,
            model_key     = "EITLEM",
            df            = df,
            pred_func     = eitlem_predictions,
            requires_cols = ["Substrate"],
            output_col    = out_col,
            handle_long   = job.handle_long_sequences,
            extra_kwargs  = {"kinetics_type": kin_flag},
        )

        job.status = "Completed"; job.completion_time = timezone.now(); job.save()

    except Exception as e:
        job.status = "Failed"; job.error_message = str(e); job.completion_time = timezone.now(); job.save()

# ------------------------------------------------------------ UniKP
@shared_task
def run_unikp_predictions(public_id):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()

    try:
        df = pd.read_csv(
            os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv")
        )

        # Decide column name & kinetics flag once
        kin_flag   = job.prediction_type.upper()          # “KCAT” | “KM”
        out_col    = "kcat (1/s)" if kin_flag == "KCAT" else "KM (mM)"

        run_model(
            job           = job,
            model_key     = "UniKP",
            df            = df,
            pred_func     = unikp_predictions,
            requires_cols = ["Substrate"],
            output_col    = out_col,
            handle_long   = job.handle_long_sequences,
            extra_kwargs  = {"kinetics_type": kin_flag},
        )

        job.status = "Completed"; job.completion_time = timezone.now(); job.save()

    except Exception as e:
        job.status = "Failed"; job.error_message = str(e); job.completion_time = timezone.now(); job.save()

# ------------------------------------------------------------ Run Both
@shared_task
def run_both_predictions(public_id, kcat_method, km_method):
    from .models import Job
    import os
    import pandas as pd
    from django.utils import timezone
    from django.conf import settings
    from api.dlkcat import dlkcat_predictions
    from api.turnup import turnup_predictions
    from api.eitlem import eitlem_predictions
    from api.unikp import unikp_predictions

    job = Job.objects.get(public_id=public_id)
    job_dir = os.path.join(settings.MEDIA_ROOT, 'jobs', str(job.public_id))
    input_file_path = os.path.join(job_dir, 'input.csv')

    # Update job status to 'Processing'
    job.status = 'Processing'
    job.predictions_made = 0
    job.total_predictions = 0
    job.save()
    multisub = False

    try:
        # Read the CSV input file
        df = pd.read_csv(input_file_path)
        sequences = df['Protein Sequence'].tolist()
        protein_ids = None

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
                public_id=job.public_id,
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
                public_id=job.public_id,
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
                public_id=job.public_id,
                protein_ids=protein_ids
            )
            results_df['kcat (1/s)'] = kcat_predictions
            invalid_indices.update(kcat_invalid_indices)
            multisub = True
        
        elif kcat_method == 'UniKP': 
            if 'Substrate' not in df.columns:
                raise ValueError('Missing "Substrate" column required for UniKP kcat predictions.')
            substrates = df['Substrate'].tolist()

            # Run UniKP predictions
            kcat_predictions, kcat_invalid_indices = unikp_predictions(
                sequences=sequences,
                substrates=substrates,
                public_id=job.public_id,
                protein_ids=protein_ids,
                kinetics_type='KCAT'
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

        # If multisubstrate format, augment KM input
        if multisub:
            augmented_rows = []  # For storing (original_idx, sequence, substrate)
            for idx, row in df.iterrows():
                smi_list = [s.strip() for s in str(row['Substrates']).split(';') if s.strip()]
                for smi in smi_list:
                    augmented_rows.append({
                        'original_idx': idx,
                        'sequence': row['Protein Sequence'],
                        'substrate': smi
                    })
            # Create augmented DataFrame
            augmented_df = pd.DataFrame(augmented_rows)
            aug_sequences = augmented_df['sequence'].tolist()
            aug_substrates = augmented_df['substrate'].tolist()

        # Run KM predictions
        if km_method == 'EITLEM':
            if not multisub and 'Substrate' not in df.columns:
                raise ValueError('Missing "Substrate" column required for EITLEM KM predictions.')
            sequences_km = aug_sequences if multisub else sequences
            substrates_km = aug_substrates if multisub else df['Substrate'].tolist()

            # Run EITLEM KM predictions
            km_predictions, km_invalid_indices = eitlem_predictions(
                sequences=sequences_km,
                substrates=substrates_km,
                public_id=job.public_id,
                protein_ids=protein_ids,
                kinetics_type='KM'
            )
            invalid_indices.update(km_invalid_indices)
        elif km_method == 'UniKP':
            if not multisub and 'Substrate' not in df.columns:
                raise ValueError('Missing "Substrate" column required for UniKP KM predictions.')
            sequences_km = aug_sequences if multisub else sequences
            substrates_km = aug_substrates if multisub else df['Substrate'].tolist()
            # Run UniKP KM predictions  
            km_predictions, km_invalid_indices = unikp_predictions(
                sequences=sequences_km,
                substrates=substrates_km,
                public_id=job.public_id,
                protein_ids=protein_ids,
                kinetics_type='KM'
            )
            invalid_indices.update(km_invalid_indices)
        else:
            raise ValueError('Invalid KM method.')
        
        if multisub:
            # Convert predictions to semicolon-separated format per original row
            from collections import defaultdict
            km_map = defaultdict(list)
            for row_idx, pred in zip(augmented_df['original_idx'], km_predictions):
                km_map[row_idx].append(str(pred))

            km_merged = [ ';'.join(km_map[i]) for i in range(len(df)) ]
            results_df['KM (mM)'] = km_merged
        else:
            results_df['KM (mM)'] = km_predictions

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
