# api/tasks.py
from celery import shared_task
from django.conf import settings
import pandas as pd
import os
import signal
import subprocess
from django.utils import timezone
try:
    from webKinPred.config_docker import MODEL_LIMITS, SERVER_LIMIT
except ImportError:
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

def handle_ram_error(job, error_msg):
    """Handle RAM/memory related errors"""
    job.status = "Failed"
    job.error_message = error_msg
    job.completion_time = timezone.now()
    job.save(update_fields=["status", "error_message", "completion_time"])
    credit_back(job.ip_address, job.requested_rows)

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
        print('handle_long:', handle_long)
        # ------- 3. length handling -------------------------------------------
        if handle_long == "truncate":
            lens_before = [len(seq) for seq in sequences]
            sequences_proc, valid_idx = truncate_sequences(sequences, limit)
            lens_after = [len(seq) for seq in sequences_proc]
            print("Lengths before truncation:", lens_before)
            print("Lengths after truncation:", lens_after)
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

        job.save(update_fields=["output_file", "error_message"])
        return csv_out
    except Exception as e:
        credit_back(job.ip_address, job.requested_rows)
        raise e
# ------------------------------------------------------------ DLKcat
@shared_task
def run_dlkcat_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save(update_fields=["status"])
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
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
    except subprocess.CalledProcessError as e:
        # Check if it's a memory error
        if e.returncode == -9 or e.returncode == 137:  # SIGKILL (OOM killer)
            handle_ram_error(job, "DLKcat prediction terminated due to insufficient memory")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )
    except MemoryError:
        handle_ram_error(job, "DLKcat prediction ran out of memory")
    except Exception as e:
        # Check if error message contains memory-related keywords
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['memory', 'ram', 'oom', 'out of memory', 'killed', 'sigkill']):
            handle_ram_error(job, f"DLKcat prediction failed due to memory issues: {str(e)}")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )
# ------------------------------------------------------------ TurNup
@shared_task
def run_turnup_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save(update_fields=["status"])
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
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )

    except subprocess.CalledProcessError as e:
        # Check if it's a memory error
        if e.returncode == -9 or e.returncode == 137:  # SIGKILL (OOM killer)
            handle_ram_error(job, "TurNup prediction terminated due to insufficient memory")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )
    except MemoryError:
        handle_ram_error(job, "TurNup prediction ran out of memory")
    except Exception as e:
        # Check if error message contains memory-related keywords
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['memory', 'ram', 'oom', 'out of memory', 'killed', 'sigkill']):
            handle_ram_error(job, f"TurNup prediction failed due to memory issues: {str(e)}")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )

# ------------------------------------------------------------ EITLEM
@shared_task
def run_eitlem_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save(update_fields=["status"])

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

        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
    except subprocess.CalledProcessError as e:
        # Check if it's a memory error
        if e.returncode == -9 or e.returncode == 137:  # SIGKILL (OOM killer)
            handle_ram_error(job, "EITLEM prediction terminated due to insufficient memory")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )
    except MemoryError:
        handle_ram_error(job, "EITLEM prediction ran out of memory")
    except Exception as e:
        # Check if error message contains memory-related keywords
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['memory', 'ram', 'oom', 'out of memory', 'killed', 'sigkill']):
            handle_ram_error(job, f"EITLEM prediction failed due to memory issues: {str(e)}")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )

# ------------------------------------------------------------ UniKP
@shared_task
def run_unikp_predictions(public_id, experimental_results=None):
    job = Job.objects.get(public_id=public_id)
    job.status = "Processing"; job.save(update_fields=["status"])

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

        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
        )
    except subprocess.CalledProcessError as e:
        # Check if it's a memory error
        if e.returncode == -9 or e.returncode == 137:  # SIGKILL (OOM killer)
            handle_ram_error(job, "UniKP prediction terminated due to insufficient memory")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )
    except MemoryError:
        handle_ram_error(job, "UniKP prediction ran out of memory")
    except Exception as e:
        # Check if error message contains memory-related keywords
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['memory', 'ram', 'oom', 'out of memory', 'killed', 'sigkill']):
            handle_ram_error(job, f"UniKP prediction failed due to memory issues: {str(e)}")
        else:
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )

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

        # ─────────────────── 1. Handle long sequences ────────────────────
        # Get limits for both methods
        kcat_limit = min(SERVER_LIMIT, MODEL_LIMITS[kcat_method])
        km_limit = min(SERVER_LIMIT, MODEL_LIMITS[km_method])
        # Use the more restrictive limit for both predictions
        limit = min(kcat_limit, km_limit)
        
        print(f'handle_long_sequences: {job.handle_long_sequences}')
        print(f'Using limit: {limit} (kcat_limit: {kcat_limit}, km_limit: {km_limit})')
        
        # Apply sequence handling
        if job.handle_long_sequences == "truncate":
            lens_before = [len(seq) for seq in sequences]
            sequences_proc, valid_indices = truncate_sequences(sequences, limit)
            lens_after = [len(seq) for seq in sequences_proc]
            print("Lengths before truncation:", lens_before)
            print("Lengths after truncation:", lens_after)
        else:  # skip
            valid_indices = get_valid_indices(sequences, limit, mode="skip")
            sequences_proc = [sequences[i] for i in valid_indices]
        
        print(f"Processing {len(sequences_proc)} sequences out of {len(sequences)} total")
        
        # Add skipped sequences to invalid_indices if we're in skip mode
        if job.handle_long_sequences == "skip":
            all_indices = set(range(len(sequences)))
            valid_indices_set = set(valid_indices)
            skipped_indices = all_indices - valid_indices_set
            invalid_indices.update(skipped_indices)

        # ─────────────────── 2.  default meta-columns ────────────────────
        n_rows          = len(df)
        kcat_src        = [f"Prediction from {kcat_method}"] * n_rows
        km_src          = [f"Prediction from {km_method}"]   * n_rows
        kcat_extra      = [""] * n_rows
        km_extra        = [""] * n_rows

        # ─────────────────── 3.  kcat predictions  ───────────────────────
        # Initialize with empty predictions for all rows
        kcat_pred = [""] * n_rows
        
        if valid_indices:  # Only predict if we have valid sequences
            if kcat_method == "DLKcat":
                subs = [df["Substrate"].iloc[i] for i in valid_indices]
                kcat_subset, bad_subset = dlkcat_predictions(
                    sequences=sequences_proc, substrates=subs,
                    public_id=job.public_id, protein_ids=protein_ids
                )
            elif kcat_method == "EITLEM":
                subs = [df["Substrate"].iloc[i] for i in valid_indices]
                kcat_subset, bad_subset = eitlem_predictions(
                    sequences=sequences_proc, substrates=subs,
                    public_id=job.public_id, protein_ids=protein_ids,
                    kinetics_type="KCAT"
                )
            elif kcat_method == "UniKP":
                subs = [df["Substrate"].iloc[i] for i in valid_indices]
                kcat_subset, bad_subset = unikp_predictions(
                    sequences=sequences_proc, substrates=subs,
                    public_id=job.public_id, protein_ids=protein_ids,
                    kinetics_type="KCAT"
                )
            elif kcat_method == "TurNup":
                multisub      = True
                subs = [df["Substrates"].iloc[i] for i in valid_indices]
                prods = [df["Products"].iloc[i] for i in valid_indices]
                kcat_subset, bad_subset = turnup_predictions(
                    sequences=sequences_proc, substrates=subs, products=prods,
                    public_id=job.public_id, protein_ids=protein_ids
                )
            else:
                raise ValueError("Invalid kcat method")
            
            # Map predictions back to original indices
            for i, pred in zip(valid_indices, kcat_subset):
                kcat_pred[i] = pred
            # Track invalid indices from predictions
            invalid_indices.update(valid_indices[j] for j in bad_subset)

        results_df["kcat (1/s)"] = kcat_pred

        # ─────────────────── 4.  KM predictions  ─────────────────────────
        # Initialize with empty predictions for all rows
        km_pred_full = [""] * n_rows
        
        if valid_indices:  # Only predict if we have valid sequences
            if multisub:
                # explode Substrates to one-row-per-substrate for valid indices only
                aug_rows = []
                for idx in valid_indices:
                    row = df.iloc[idx]
                    for smi in [s.strip() for s in str(row["Substrates"]).split(";") if s.strip()]:
                        aug_rows.append({"original_idx": idx,
                                         "sequence": sequences_proc[valid_indices.index(idx)],  # Use processed sequence
                                         "substrate": smi})
                aug_df       = pd.DataFrame(aug_rows)
                seq_km       = aug_df["sequence"].tolist()
                subs_km      = aug_df["substrate"].tolist()
            else:
                seq_km       = sequences_proc
                subs_km      = [df["Substrate"].iloc[i] for i in valid_indices]

            if km_method == "EITLEM":
                km_pred, bad_km = eitlem_predictions(
                    sequences=seq_km, substrates=subs_km,
                    public_id=job.public_id, protein_ids=protein_ids,
                    kinetics_type="KM"
                )
            elif km_method == "UniKP":
                km_pred, bad_km = unikp_predictions(
                    sequences=seq_km, substrates=subs_km,
                    public_id=job.public_id, protein_ids=protein_ids,
                    kinetics_type="KM"
                )
            else:
                raise ValueError("Invalid KM method")

            # Track invalid indices from KM predictions
            invalid_indices.update(valid_indices[j] for j in bad_km)

            if multisub:
                # regroup semicolon-separated predictions for valid indices
                from collections import defaultdict
                km_map = defaultdict(list)
                for r, pred in zip(aug_df["original_idx"], km_pred):
                    km_map[r].append(str(pred))
                # Fill in predictions for valid indices only
                for idx in valid_indices:
                    if idx in km_map:
                        km_pred_full[idx] = ";".join(km_map[idx])
            else:
                # Map predictions back to original indices
                for i, pred in zip(valid_indices, km_pred):
                    km_pred_full[i] = pred

        results_df["KM (mM)"] = km_pred_full

        # ─────────────────── 5.  experimental overwrites ─────────────────
        if experimental_results:
            for exp in experimental_results:
                if not exp.get("found"):
                    continue
                idx   = exp["idx"]
                p_seq = exp["protein_sequence"]
                # Check against original sequences (not processed ones)
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

        # ─────────────────── 6.  attach meta-columns  ────────────────────
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

        # ─────────────────── 7.  save & update job  ──────────────────────
        out_csv = os.path.join(job_dir, "output.csv")
        results_df.to_csv(out_csv, index=False)

        # Calculate processed rows (rows with non-empty predictions for both kcat and KM)
        try:
            kcat_processed = int((results_df["kcat (1/s)"] != "").sum())
            km_processed = int((results_df["KM (mM)"] != "").sum())
            processed_rows = min(kcat_processed, km_processed)  # Conservative estimate
        except Exception:
            processed_rows = len(valid_indices) if valid_indices else 0

        # Credit back unused reactions
        to_refund = max(0, int(job.requested_rows) - processed_rows)
        if to_refund > 0:
            credit_back(job.ip_address, to_refund)

        if invalid_indices:
            invalid_indices = sorted(invalid_indices)
            job.error_message = (
                f"Predictions could not be made for {len(invalid_indices)} row(s): "
                f"{', '.join(map(str, invalid_indices))}"
            )
        else:
            job.error_message = ""
            
        Job.objects.filter(pk=job.pk).update(
            status="Completed",
            completion_time=timezone.now(),
            output_file=os.path.relpath(out_csv, settings.MEDIA_ROOT),
        )

    except subprocess.CalledProcessError as e:
        # Check if it's a memory error
        if e.returncode == -9 or e.returncode == 137:  # SIGKILL (OOM killer)
            handle_ram_error(job, "Run Both prediction terminated due to insufficient memory")
        else:
            credit_back(job.ip_address, job.requested_rows)
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(e),
                completion_time=timezone.now()
            )
    except MemoryError:
        handle_ram_error(job, "Run Both prediction ran out of memory")
    except Exception as exc:
        # Check if error message contains memory-related keywords
        error_str = str(exc).lower()
        if any(keyword in error_str for keyword in ['memory', 'ram', 'oom', 'out of memory', 'killed', 'sigkill']):
            handle_ram_error(job, f"Run Both prediction failed due to memory issues: {str(exc)}")
        else:
            credit_back(job.ip_address, job.requested_rows)
            Job.objects.filter(pk=job.pk).update(
                status="Failed",
                error_message=str(exc),
                completion_time=timezone.now()
            )
