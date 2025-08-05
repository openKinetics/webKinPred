import json
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from api.utils.convert_to_mol import convert_to_mol
import csv
import subprocess
import tempfile
import os
import shutil
from webKinPred.config_local import CONDA_PATH,TARGET_DBS
 
@csrf_exempt
def validate_input(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)

    file = request.FILES.get('file')
    kcat_method = request.POST.get('kcatMethod')
    km_method = request.POST.get('kmMethod')
    if not file:
        return JsonResponse({'error': 'No file provided.'}, status=400)

    try:
        df = pd.read_csv(file)
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
    length_violations = {
        'EITLEM': 0,
        'TurNup': 0,
        'UniKP': 0,
        'DLKcat': 0,
        'Server': 0
    }
    # Validate Substrates
    if (km_method in ['EITLEM','UniKP']) or (kcat_method in ['DLKcat', 'EITLEM','UniKP']) and 'Substrate' in df.columns:
        for i, val in enumerate(df['Substrate']):
            mol = convert_to_mol(val)
            if mol is None:
                invalid_substrates.append({'row': i + 1, 'value': val, 'reason': 'Invalid SMILES/InChI'})

    elif kcat_method == 'TurNup' and 'Substrates' in df.columns and 'Products' in df.columns:
        for i, (subs, prods) in enumerate(zip(df['Substrates'], df['Products'])):
            all_valid = True
            for s in subs.split(';'):
                if convert_to_mol(s.strip()) is None:
                    all_valid = False
            for p in prods.split(';'):
                if convert_to_mol(p.strip()) is None:
                    all_valid = False
            if not all_valid:
                invalid_substrates.append({'row': i + 1, 'value': f"{subs} => {prods}", 'reason': 'Invalid substrate/product SMILES/InChI'})

    # Validate Proteins
    if 'Protein Sequence' in df.columns:
        for i, seq in enumerate(df['Protein Sequence']):
            if not isinstance(seq, str) or len(seq.strip()) == 0:
                invalid_proteins.append({'row': i + 1, 'reason': 'Empty sequence'})
                continue
            seq_len = len(seq)
            if seq_len > server_limit:
                length_violations['Server'] += 1
            if seq_len > model_limits['EITLEM']:
                length_violations['EITLEM'] += 1
            if seq_len > model_limits['TurNup']:
                length_violations['TurNup'] += 1
            if seq_len > model_limits['UniKP']:
                length_violations['UniKP'] += 1
            if seq_len > model_limits['DLKcat']:
                length_violations['DLKcat'] += 1
            if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in seq.upper()):
                invalid_proteins.append({'row': i + 1, 'reason': 'Invalid characters in sequence'})

    return JsonResponse({
        'invalid_substrates': invalid_substrates,
        'invalid_proteins': invalid_proteins,
        'protein_similarity': [],  # placeholder for future
        'length_violations': length_violations,
    }, status=200)

@csrf_exempt
def sequence_similarity_summary(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
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
        result = calculate_sequence_similarity_by_histogram(sequences)
        return JsonResponse(result, status=200)

    except Exception as e:
        print(e)
        return JsonResponse({'error': str(e)}, status=500)
    

def calculate_sequence_similarity_by_histogram(
    input_sequences: list[str]
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
    # Pre-created target databases (assumed built and stored already).
    target_dbs = TARGET_DBS
    # Write input sequences to a temporary FASTA file.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as query_file:
        query_file_path = query_file.name
        for idx, seq in enumerate(input_sequences, start=1):
            query_file.write(f">seq{idx}\n")
            query_file.write(seq + "\n")
    
    # Create the query database once (using the pre-created query FASTA).
    temp_query_dir = tempfile.mkdtemp()
    query_db = os.path.join(temp_query_dir, "queryDB")
    subprocess.run(
        [CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", "createdb", query_file_path, query_db],
        check=True
    )
    
    def run_mmseqs_search_with_precreated_query(query_db: str, target_db: str, query_file_path: str) -> tuple[dict, dict]:
        """
        Returns:
        - max_identity: max % identity per sequence
        - mean_identity: mean % identity per sequence
        """
        tmp_dir = tempfile.mkdtemp()
        result_db = os.path.join(tmp_dir, "resultDB")
        result_file = os.path.join(tmp_dir, "result.m8")

        try:
            subprocess.run(
                [   CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", "search",
                    query_db, target_db, result_db, tmp_dir,
                    "--max-seqs", "5000", "-s", "7.5",
                    "-e", "0.1"
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            subprocess.run(
                [CONDA_PATH, "run", "-n", "mmseqs2_env", "mmseqs", "convertalis",
                query_db, target_db, result_db, result_file,
                "--format-output", "query,target,pident"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )

            max_identity = {}
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

                        if query_id not in max_identity or pident > max_identity[query_id]:
                            max_identity[query_id] = pident
                        identity_lists.setdefault(query_id, []).append(pident)

            # Assign 0.0 if no hit
            with open(query_file_path, "r") as f:
                for line in f:
                    if line.startswith(">"):
                        qid = line[1:].strip()
                        if qid not in max_identity:
                            max_identity[qid] = 0.0
                        if qid not in identity_lists:
                            identity_lists[qid] = [0.0]

            mean_identity = {k: sum(v)/len(v) for k, v in identity_lists.items()}
            return max_identity, mean_identity

        finally:
            shutil.rmtree(tmp_dir)

    # Prepare the final histogram dictionary.
    # For each method, create a histogram that maps each integer identity (as a string) to the percentage frequency.
    method_histograms = {}
    
    for method, target_db in target_dbs.items():
        query_to_max, query_to_mean = run_mmseqs_search_with_precreated_query(query_db, target_db, query_file_path)
        total_seqs = len(query_to_max)

        # Histogram based on max identities
        histogram_max = {str(i): 0 for i in range(101)}
        for identity in query_to_max.values():
            identity_int = int(round(identity))
            identity_int = max(0, min(100, identity_int))
            histogram_max[str(identity_int)] += 1
        histogram_max_perc = {k: (v / total_seqs * 100) if total_seqs else 0.0 for k, v in histogram_max.items()}

        # Histogram based on mean identities
        histogram_mean = {str(i): 0 for i in range(101)}
        for identity in query_to_mean.values():
            identity_int = int(round(identity))
            identity_int = max(0, min(100, identity_int))
            histogram_mean[str(identity_int)] += 1
        histogram_mean_perc = {k: (v / total_seqs * 100) if total_seqs else 0.0 for k, v in histogram_mean.items()}

        method_histograms[method] = {
            "histogram_max": histogram_max_perc,
            "histogram_mean": histogram_mean_perc,
            "average_max_similarity": round(sum(query_to_max.values()) / total_seqs, 2),
            "average_mean_similarity": round(sum(query_to_mean.values()) / total_seqs, 2),
            "count_max": histogram_max, 
            "count_mean": histogram_mean  
        }
    # Clean up temporary files.
    os.remove(query_file_path)
    shutil.rmtree(temp_query_dir)

    return method_histograms
