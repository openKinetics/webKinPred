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
from api.utils.safe_read import safe_read_csv
from api.dlkcat import dlkcat_predictions
from api.turnup import turnup_predictions
from api.eitlem import eitlem_predictions
from api.unikp import unikp_predictions
from api.utils.quotas import credit_back
from api.utils.extra_info import build_extra_info, _source

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
    experimental_results: dict | None = None
):
    """
    Generic prediction runner.

    Parameters
    ----------
    job             : Job   - Django Job row (already fetched)
    model_key       : str   - key inside MODEL_LIMITS (e.g. 'dlkcat')
    df              : DataFrame of the CSV
    pred_func       : callable(**kwargs) -> (preds, invalid_idxs)
    requires_cols   : list  - columns that must exist in df
    extra_kwargs    : dict  - pass-through to pred_func (e.g. kinetics_type)
    output_col      : str   - column to write predictions into
    handle_long     : 'skip'|'truncate'
    experimental_results : dict | None - If provided, will be used instead of predictions
    """

    try:
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
        extra_info = ["" for _ in sequences]
        sources = ["" for _ in sequences]  # For storing source info if needed
        invalid_global: list[int] = []
        if valid_idx:
            pred_subset, invalid_subset = pred_func(**kwargs)
            for i, p in zip(valid_idx, pred_subset):
                full_preds[i] = p
                sources [i] = f"Prediction from {model_key}"  
            invalid_global = [valid_idx[i] for i in invalid_subset]
        exp_res_key = 'km_value' if job.prediction_type == 'Km' else 'kcat_value'
        experimental_results = experimental_results or []
        for exp_res in experimental_results:
            if exp_res['found']:
                idx = exp_res['idx']
                if exp_res['protein_sequence'] != sequences[idx]:
                    print(
                        f"Protein sequence mismatch at index {idx}: "
                        f"expected {sequences[idx]}, got {exp_res['protein_sequence']}"
                    )
                    continue
                prediction = full_preds[idx]
                full_preds[idx] = exp_res[exp_res_key]
                sources[idx] = _source(exp_res)
                extra_info[idx] = build_extra_info(exp_res, job.prediction_type, prediction, model_key)
        
        # ------- 6. write CSV --------------------------------------------------
        df.insert(0, "Extra Info", extra_info)
        df.insert(0, "Source", sources)
        df.insert(0, output_col, full_preds)
        csv_out = os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "output.csv")
        df.to_csv(csv_out, index=False)
        # ------- 7. process results and credit back --------------------------
        try:
            processed_rows = int((df[output_col] != "").sum())   # non-empty predictions
        except Exception:
            processed_rows = 0 
        # Credit back unused reactions
        to_refund = max(0, int(job.requested_rows) - processed_rows)
        credit_back(job.ip_address, to_refund)
        # ------- 7. update job -------------------------------------------------
        job.output_file.name = os.path.relpath(csv_out, settings.MEDIA_ROOT)
        job.error_message = (
            f"Predictions could not be made for {len(invalid_global)} row(s): "
            + ", ".join(map(str, invalid_global))
        ) if invalid_global else ""
        return csv_out
    except Exception as e:
        credit_back(job.ip_address, job.requested_rows)
        raise e
# ------------------------------------------------------------ DLKcat
@shared_task
def run_dlkcat_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()
    try:
        df = safe_read_csv(
            os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv"),
            job.ip_address,
            job.requested_rows
        )
        if df is None:
            raise ValueError("Failed to read input CSV file.")
        run_model(
            job            = job,
            model_key      = "DLKcat",
            df             = df,
            pred_func      = dlkcat_predictions,
            requires_cols  = ["Substrate"],
            output_col     = "kcat (1/s)",
            handle_long    = job.handle_long_sequences,
            experimental_results = experimental_results
        )
        job.status = "Completed"
        job.completion_time = timezone.now()
        job.save()
    except Exception as e:
        job.status = "Failed"; job.error_message = str(e); job.completion_time = timezone.now(); job.save()
# ------------------------------------------------------------ TurNup
@shared_task
def run_turnup_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()
    try:
        df = safe_read_csv(
            os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv"),
            job.ip_address,
            job.requested_rows
        )
        if df is None:
            raise ValueError("Failed to read input CSV file.")
        run_model(
            job           = job,
            model_key     = "TurNup",
            df            = df,
            pred_func     = turnup_predictions,
            requires_cols = ["Substrates", "Products"],
            output_col    = "kcat (1/s)",
            handle_long   = job.handle_long_sequences,
            experimental_results = experimental_results
        )
        job.status = "Completed"; job.completion_time = timezone.now(); job.save()

    except Exception as e:
        job.status = "Failed"; job.error_message = str(e); job.completion_time = timezone.now(); job.save()

# ------------------------------------------------------------ EITLEM
@shared_task
def run_eitlem_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()

    try:
        df = safe_read_csv(
            os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv"),
            job.ip_address,
            job.requested_rows
        )
        if df is None:
            raise ValueError("Failed to read input CSV file.")
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
            experimental_results = experimental_results
        )

        job.status = "Completed"; job.completion_time = timezone.now(); job.save()

    except Exception as e:
        job.status = "Failed"; job.error_message = str(e); job.completion_time = timezone.now(); job.save()

# ------------------------------------------------------------ UniKP
@shared_task
def run_unikp_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save()

    try:
        df = safe_read_csv(
            os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id), "input.csv"),
            job.ip_address,
            job.requested_rows
        )
        if df is None:
            raise ValueError("Failed to read input CSV file.")

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
            experimental_results = experimental_results
        )

        job.status = "Completed"; job.completion_time = timezone.now(); job.save()

    except Exception as e:
        job.status = "Failed"; job.error_message = str(e); job.completion_time = timezone.now(); job.save()

# ------------------------------------------------------------ Run Both
# ------------------------------------------------------------ Run Both
@shared_task
def run_both_predictions(public_id, experimental_results=None):
    """
    Predict kcat **and** KM for every row, optionally overwriting either
    (or both) with experimental values supplied via `experimental_results`.

    experimental_results is a flat list whose items have at least:
        {'found': bool, 'idx': <row>, 'kcat_value' or 'km_value', …}
    For `lookup_experimental(..., param_type="both")` the list is
    interleaved:  kcat₀, KM₀, kcat₁, KM₁, …
    """
    from collections import defaultdict
    from .models import Job
    import os, pandas as pd
    from django.conf import settings
    from django.utils import timezone
    from api.dlkcat import dlkcat_predictions
    from api.turnup import turnup_predictions
    from api.eitlem import eitlem_predictions
    from api.unikp import unikp_predictions
    from api.utils.extra_info import build_extra_info, _source
    from api.utils.quotas import credit_back

    # ───────────────────────── 0.  House-keeping ─────────────────────────
    job      = Job.objects.get(public_id=public_id)
    job_dir  = os.path.join(settings.MEDIA_ROOT, "jobs", str(job.public_id))
    infile   = os.path.join(job_dir, "input.csv")

    job.status, job.predictions_made, job.total_predictions = "Processing", 0, 0
    job.save(update_fields=["status", "predictions_made", "total_predictions"])

    kcat_method = job.kcat_method
    km_method   = job.km_method

    multisub = False          # only true for TurNup
    invalid_indices = set()   # will accumulate across both passes

    try:
        df              = pd.read_csv(infile)
        sequences       = df["Protein Sequence"].tolist()
        results_df      = df.copy()        # will receive new columns
        protein_ids     = None             # placeholder for future use

        # ─────────────────── 1.  default meta-columns ────────────────────
        n_rows          = len(df)
        kcat_src        = [f"Prediction from {kcat_method}"] * n_rows
        km_src          = [f"Prediction from {km_method}"]   * n_rows
        kcat_extra      = [""] * n_rows
        km_extra        = [""] * n_rows

        # ─────────────────── 2.  kcat predictions  ───────────────────────
        if kcat_method == "DLKcat":
            subs = df["Substrate"].tolist()
            kcat_pred, bad = dlkcat_predictions(
                sequences=sequences, substrates=subs,
                public_id=job.public_id, protein_ids=protein_ids
            )
        elif kcat_method == "EITLEM":
            subs = df["Substrate"].tolist()
            kcat_pred, bad = eitlem_predictions(
                sequences=sequences, substrates=subs,
                public_id=job.public_id, protein_ids=protein_ids,
                kinetics_type="KCAT"
            )
        elif kcat_method == "UniKP":
            subs = df["Substrate"].tolist()
            kcat_pred, bad = unikp_predictions(
                sequences=sequences, substrates=subs,
                public_id=job.public_id, protein_ids=protein_ids,
                kinetics_type="KCAT"
            )
        elif kcat_method == "TurNup":
            multisub      = True
            subs, prods   = df["Substrates"].tolist(), df["Products"].tolist()
            kcat_pred, bad = turnup_predictions(
                sequences=sequences, substrates=subs, products=prods,
                public_id=job.public_id, protein_ids=protein_ids
            )
        else:
            raise ValueError("Invalid kcat method")

        results_df["kcat (1/s)"] = kcat_pred
        invalid_indices.update(bad)

        # ─────────────────── 3.  KM predictions  ─────────────────────────
        if multisub:
            # explode Substrates to one-row-per-substrate
            aug_rows = []
            for idx, row in df.iterrows():
                for smi in [s.strip() for s in str(row["Substrates"]).split(";") if s.strip()]:
                    aug_rows.append({"original_idx": idx,
                                     "sequence": row["Protein Sequence"],
                                     "substrate": smi})
            aug_df       = pd.DataFrame(aug_rows)
            seq_km       = aug_df["sequence"].tolist()
            subs_km      = aug_df["substrate"].tolist()
        else:
            seq_km       = sequences
            subs_km      = df["Substrate"].tolist()

        if km_method == "EITLEM":
            km_pred, bad = eitlem_predictions(
                sequences=seq_km, substrates=subs_km,
                public_id=job.public_id, protein_ids=protein_ids,
                kinetics_type="KM"
            )
        elif km_method == "UniKP":
            km_pred, bad = unikp_predictions(
                sequences=seq_km, substrates=subs_km,
                public_id=job.public_id, protein_ids=protein_ids,
                kinetics_type="KM"
            )
        else:
            raise ValueError("Invalid KM method")

        invalid_indices.update(bad)

        if multisub:
            # regroup semicolon-separated predictions
            from collections import defaultdict
            km_map = defaultdict(list)
            for r, pred in zip(aug_df["original_idx"], km_pred):
                km_map[r].append(str(pred))
            results_df["KM (mM)"] = [";".join(km_map[i]) for i in range(n_rows)]
        else:
            results_df["KM (mM)"] = km_pred

        # ─────────────────── 4.  experimental overwrites ─────────────────
        if experimental_results:
            for exp in experimental_results:
                if not exp.get("found"):
                    continue
                idx   = exp["idx"]
                p_seq = exp["protein_sequence"]
                if p_seq != sequences[idx]:
                    print(
                        f"Protein sequence mismatch at index {idx}: "
                        f"expected {sequences[idx]}, got {p_seq}"
                    )
                    continue

                if "kcat_value" in exp:          # kcat overwrite
                    prev_val               = results_df.at[idx, "kcat (1/s)"]
                    results_df.at[idx, "kcat (1/s)"] = exp["kcat_value"]
                    kcat_src[idx]          = _source(exp)
                    kcat_extra[idx]        = build_extra_info(
                        exp, "kcat", prev_val, kcat_method
                    )
                elif "km_value" in exp:          # KM overwrite
                    prev_val               = results_df.at[idx, "KM (mM)"]
                    results_df.at[idx, "KM (mM)"] = exp["km_value"]
                    km_src[idx]            = _source(exp)
                    km_extra[idx]          = build_extra_info(
                        exp, "Km", prev_val, km_method
                    )

        # ─────────────────── 5.  attach meta-columns  ────────────────────
        results_df.insert(0, "Extra Info KM",   km_extra)
        results_df.insert(0, "Source KM",       km_src)
        results_df.insert(0, "Extra Info kcat", kcat_extra)
        results_df.insert(0, "Source kcat",     kcat_src)

        # bring prediction columns to the front
        preferred_order = [
            "kcat (1/s)", "Source kcat", "Extra Info kcat",
            "KM (mM)",    "Source KM",   "Extra Info KM"
        ]
        results_df = results_df[
            preferred_order + [c for c in results_df.columns if c not in preferred_order]
        ]

        # ─────────────────── 6.  save & update job  ──────────────────────
        out_csv = os.path.join(job_dir, "output.csv")
        results_df.to_csv(out_csv, index=False)

        if invalid_indices:
            invalid_indices = sorted(invalid_indices)
            job.error_message = (
                f"Predictions could not be made for {len(invalid_indices)} row(s): "
                f"{', '.join(map(str, invalid_indices))}"
            )
            credit_back(job.ip_address,
                        max(0, job.requested_rows - len(invalid_indices)))
        else:
            job.error_message = ""

        job.status          = "Completed"
        job.completion_time = timezone.now()
        job.output_file.name = os.path.relpath(out_csv, settings.MEDIA_ROOT)
        job.save()

    except Exception as exc:
        credit_back(job.ip_address, job.requested_rows)
        job.status          = "Failed"
        job.error_message   = str(exc)
        job.completion_time = timezone.now()
        job.save()
