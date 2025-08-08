import json
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
from api.utils.convert_to_mol import convert_to_mol
import csv
import subprocess
import tempfile
import os
import shutil
from webKinPred.config_local import CONDA_PATH,TARGET_DBS
from api.progress import (
    push_line, finish_session, sse_generator,
    cancel_session, is_cancelled, get_pid_key,
    redis_conn
)
from api.log_sanitiser import sanitise_log_line

@csrf_exempt
def progress_stream(request):
    """
    SSE endpoint: /api/progress-stream/?session_id=XYZ
    Streams logs for the given session_id.
    """
    session_id = request.GET.get("session_id")
    if not session_id:
        return JsonResponse({"error": "session_id query param required"}, status=400)
    # The start_session(session_id) call is REMOVED. It is no longer needed.
    response = StreamingHttpResponse(
        streaming_content=sse_generator(session_id),
        content_type="text/event-stream",
    )
    # Helpful SSE headers
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  
    return response

def _run_and_stream(cmd, session_id: str, cwd: str | None = None, env: dict | None = None, fail_ok=False):
    echoed = "$ " + " ".join(cmd)
    san_line = sanitise_log_line(echoed, TARGET_DBS)
    push_line(session_id, san_line)

    pid_key = get_pid_key(session_id)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid  
        )
        # Store the PID in Redis with a 15-minute expiry as a safety net
        redis_conn.set(pid_key, proc.pid, ex=900)
        for raw in proc.stdout:
            raw = raw.rstrip("\n")
            # The is_cancelled check is now a secondary guard
            if is_cancelled(session_id):
                break
            safe = sanitise_log_line(raw, TARGET_DBS)
            push_line(session_id, safe)
        rc = proc.wait()
    finally:
        if proc:
            print(f"[cleanup] Deleting PID key for session {session_id}")
            redis_conn.delete(pid_key)

    if is_cancelled(session_id):
        print(f"[run_and_stream] Step for session {session_id} was cancelled.")
        return
    if rc != 0 and not fail_ok:
        push_line(session_id, f"[ERROR] Command failed with exit code {rc}")
        raise subprocess.CalledProcessError(rc, cmd)
    elif rc != 0 and fail_ok:
        push_line(session_id, f"[WARN] Command returned non-zero exit code {rc} (continuing)")
    else:
        push_line(session_id, "[OK] Completed")

@csrf_exempt
def cancel_validation(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    session_id = request.POST.get("session_id")
    if not session_id:
        return JsonResponse({"error": "session_id required"}, status=400)
    ok = cancel_session(session_id)
    return JsonResponse({"ok": bool(ok)})

@csrf_exempt
def validate_input(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)

    file = request.FILES.get('file')
    if not file:
        return JsonResponse({'error': 'No file provided.'}, status=400)

    try:
        df = pd.read_csv(file)
        df = df.dropna(how='all')  # Remove empty rows
    except Exception as e:
        return JsonResponse({'error': f'Invalid CSV format: {str(e)}'}, status=400)

    invalid_substrates = []
    invalid_proteins = []

    # Define sequence length limits for models
    model_limits = {
        'EITLEM': 1024,
        'TurNup': 1024,
        'UniKP': 1000,
        'DLKcat': float('inf'),  # no limit
    }
    server_limit = 1500  # server-wide max length
    length_violations = {'EITLEM': 0, 'TurNup': 0, 'UniKP': 0, 'DLKcat': 0, 'Server': 0}

    def convert_to_mol_safe(x):
        # Robust guard for NaN / non-str / empty strings
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        if not isinstance(x, str):
            return None
        s = x.strip()
        if not s:
            return None
        return convert_to_mol(s)

    # Validate Substrates
    if 'Substrate' in df.columns:
        val_to_output = {}
        # Single-substrate schema
        for i, val in enumerate(df['Substrate']):
            if val in val_to_output:
                invalid_substrates.append(val_to_output[val])
            if convert_to_mol_safe(val) is None:
                val_to_output[val] = {
                    'row': i + 1,
                    'value': val,  # offending SMILES/InChI
                    'reason': 'Invalid SMILES/InChI'
                }
                invalid_substrates.append(val_to_output[val])

    elif ('Substrates' in df.columns and 'Products' in df.columns):
        # Multi-substrate schema
        for i, (subs, prods) in enumerate(zip(df['Substrates'], df['Products'])):
            invalid_tokens = []

            # Validate each substrate token
            for s in str(subs).split(';'):
                tok = s.strip()
                if tok and convert_to_mol_safe(tok) is None:
                    invalid_tokens.append(tok)

            # Validate each product token
            for p in str(prods).split(';'):
                tok = p.strip()
                if tok and convert_to_mol_safe(tok) is None:
                    invalid_tokens.append(tok)

            if invalid_tokens:
                invalid_substrates.append({
                    'row': i + 1,
                    'value': invalid_tokens,  # list of offending tokens
                    'reason': 'Invalid substrate/product SMILES/InChI'
                })
    else:
        return JsonResponse(
            {'error': 'CSV must contain "Substrate" column OR "Substrates" and "Products" columns'},
            status=400
        )

    # Validate Proteins
    if 'Protein Sequence' in df.columns:
        alphabet = set('ACDEFGHIKLMNPQRSTVWY')
        for i, seq in enumerate(df['Protein Sequence']):
            rownum = i + 1
            if not isinstance(seq, str) or len(seq.strip()) == 0:
                invalid_proteins.append({'row': rownum, 'value': seq or '', 'reason': 'Empty sequence'})
                continue

            seq = seq.strip()
            seq_len = len(seq)
            if seq_len > server_limit:
                length_violations['Server'] += 1
            if seq_len > model_limits['EITLEM']:
                length_violations['EITLEM'] += 1
            if seq_len > model_limits['TurNup']:
                length_violations['TurNup'] += 1
            if seq_len > model_limits['UniKP']:
                length_violations['UniKP'] += 1
            # DLKcat is inf; kept for symmetry
            if seq_len > model_limits['DLKcat']:
                length_violations['DLKcat'] += 1

            invalid_chars = sorted({c for c in seq if c not in alphabet})
            if invalid_chars:
                invalid_proteins.append({
                    'row': rownum,
                    'value': seq,               # offending sequence
                    'invalid_chars': invalid_chars,
                    'reason': 'Invalid characters in sequence'
                })
    else:
        return JsonResponse({'error': 'CSV must contain a "Protein Sequence" column'}, status=400)

    return JsonResponse({
        'invalid_substrates': invalid_substrates,
        'invalid_proteins': invalid_proteins,
        'protein_similarity': [],  # placeholder
        'length_violations': length_violations,
    }, status=200)

@csrf_exempt
def sequence_similarity_summary(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    session_id = request.POST.get('validationSessionId') or "default"
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'CSV file not provided'}, status=400)

        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)
        if 'Protein Sequence' not in df.columns:
            return JsonResponse({'error': 'CSV must contain a "Protein Sequence" column'}, status=400)

        sequences = df['Protein Sequence'].dropna().tolist()
        if not sequences:
            return JsonResponse({'error': 'No valid protein sequences found in CSV'}, status=400)
        push_line(session_id, "==> Starting MMseqs2 similarity analysis")
        result = calculate_sequence_similarity_by_histogram(sequences, session_id=session_id)
        push_line(session_id, "==> Similarity histograms computed successfully")
        return JsonResponse(result, status=200)

    except Exception as e:
        push_line(session_id, f"[EXCEPTION] {e}")
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        finish_session(session_id)
    
def calculate_sequence_similarity_by_histogram(
    input_sequences: list[str],
    session_id: str = "default"
) -> dict:
    """ 
    Computes the distribution histogram of maximum sequence identity of user-provided protein 
    sequences relative to pre-created training target databases (for DLKcat, TurNup, and EITLEM)
    by leveraging the mmseqs2 tool. For each input sequence, mmseqs2 is used to search against a
    target database associated with each prediction method, and the highest percent identity is obtained.
    
    Rather than grouping by fixed identity ranges, this function rounds each maximum identity to the nearest
    integer (0-100) and computes the percentage frequency for each identity value.

    Parameters
    ----------
    input_sequences : list of str
        The protein sequences provided by the user.

    Returns
    -------
    dict
        A dictionary with keys corresponding to the prediction methods ("DLKcat", "TurNup", "EITLEM").
        Each key maps to a sub-dictionary where each key is an integer (as a string) representing a percent identity (0 to 100),
        and the value is the percentage of input sequences that have that rounded identity value.
    """
    target_dbs = TARGET_DBS

    unique_sequences = list(dict.fromkeys(input_sequences))
    seq_to_unique_id = {seq: f"useq{idx}" for idx, seq in enumerate(unique_sequences)}
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta", dir="/home/saleh/mmseqs_tmp") as query_file:
        query_file_path = query_file.name
        for seq, unique_id in seq_to_unique_id.items():
            query_file.write(f">{unique_id}\n{seq}\n")

    temp_query_dir = tempfile.mkdtemp(dir="/home/saleh/mmseqs_tmp")
    query_db = os.path.join(temp_query_dir, "queryDB")

    _run_and_stream(
        [CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", "createdb", query_file_path, query_db],
        session_id=session_id
    )

    def run_mmseqs_search_with_precreated_query(query_db: str, target_db: str, query_file_path: str, method_name: str) -> tuple[dict, dict]:
        tmp_dir = tempfile.mkdtemp(dir="/home/saleh/mmseqs_tmp")
        result_db = os.path.join(tmp_dir, "resultDB")
        result_file = os.path.join(tmp_dir, "result.m8")

        try:
            push_line(session_id, f"--> [{method_name}] Running search")
            _run_and_stream(
                [
                    CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", "search",
                    query_db, target_db, result_db, tmp_dir,
                    "--max-seqs", "5000", "-s", "7.5", 
                    "-e", "0.001", "-v", "0",
                ],
                session_id=session_id,
                fail_ok=True
            )

            push_line(session_id, f"--> [{method_name}] Converting alignments")
            _run_and_stream(
                [
                    CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", "convertalis",
                    query_db, target_db, result_db, result_file,
                    "--format-output", "query,target,pident"
                ],
                session_id=session_id,
                fail_ok=True
            )
            identity_lists = {}
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    for line in f:
                        fields = line.strip().split("\t")
                        if len(fields) < 3:
                            continue
                        query_id = fields[0]
                        try:
                            pident = float(fields[2])
                        except ValueError:
                            continue
                        identity_lists.setdefault(query_id, []).append(pident)

            with open(query_file_path, "r") as f:
                for line in f:
                    if line.startswith(">"):
                        qid = line[1:].strip()
                        if qid not in identity_lists:
                            identity_lists[qid] = [0.0]
            mean_identity = {k: (sum(v) / len(v)) for k, v in identity_lists.items()}
            max_identity = {k: max(v) for k, v in identity_lists.items()}
            push_line(session_id, f"--> [{method_name}] Aggregated {len(max_identity)} sequences")
            return max_identity, mean_identity
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    method_histograms = {}
    for method, target_db in target_dbs.items():
        push_line(session_id, f"==> Processing DB: {method}")
        unique_query_to_max, unique_query_to_mean = run_mmseqs_search_with_precreated_query(query_db, target_db, query_file_path, method)

        query_to_max = {}
        query_to_mean = {}
        for original_idx, seq in enumerate(input_sequences):
            original_seq_id = f"seq{original_idx + 1}"
            unique_id = seq_to_unique_id[seq]
            query_to_max[original_seq_id] = unique_query_to_max.get(unique_id, 0.0)
            query_to_mean[original_seq_id] = unique_query_to_mean.get(unique_id, 0.0)
        total_seqs = len(query_to_max)
        histogram_max = {str(i): 0 for i in range(101)}
        for identity in query_to_max.values():
            rounded = int(round(identity))
            rounded= max(0, min(100, rounded))
            histogram_max[str(rounded)] += 1
            
        histogram_max_perc = {k: (v / total_seqs * 100) if total_seqs else 0.0 for k, v in histogram_max.items()}
        histogram_mean = {str(i): 0 for i in range(101)}
        for identity in query_to_mean.values():
            rounded = int(round(identity))
            rounded = max(0, min(100, rounded))
            histogram_mean[str(rounded)] += 1
        histogram_mean_perc = {k: (v / total_seqs * 100) if total_seqs else 0.0 for k, v in histogram_mean.items()}

        method_histograms[method] = {
            "histogram_max": histogram_max_perc,
            "histogram_mean": histogram_mean_perc,
            "average_max_similarity": round(sum(query_to_max.values()) / total_seqs * 100, 2) if total_seqs else 0.0,
            "average_mean_similarity": round(sum(query_to_mean.values()) / total_seqs * 100, 2) if total_seqs else 0.0,
            "count_max": histogram_max,
            "count_mean": histogram_mean
        }

    try:
        os.remove(query_file_path)
    except Exception:
        pass
    shutil.rmtree(temp_query_dir, ignore_errors=True)
    return method_histograms