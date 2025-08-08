import os
import subprocess
import pandas as pd
from rdkit import Chem
from api.utils.convert_to_mol import convert_to_mol
from api.models import Job
from webKinPred.settings import MEDIA_ROOT
from webKinPred.config_local import PYTHON_PATHS, PREDICTION_SCRIPTS

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
        # Log all stdout lines
        for line in process.stdout:
            print("[UniKP subprocess]", line.strip())
        # Wait for the subprocess to finish
        process.wait()

        if process.returncode != 0:
            # An error occurred
            raise subprocess.CalledProcessError(process.returncode, process.args)

    except Exception as e:
        print("An error occurred while running the subprocess:")
        print(e)
        raise e

def unikp_predictions(sequences, substrates, public_id, protein_ids=None, kinetics_type='KCAT'):
    print("Running UniKP model...")

    # Get the Job object
    job = Job.objects.get(public_id=public_id)

    # Initialize progress fields
    job.molecules_processed = 0
    job.invalid_molecules = 0
    job.predictions_made = 0
    job.save(update_fields=["molecules_processed", "invalid_molecules", "predictions_made"])

    # Define paths
    python_path = PYTHON_PATHS['UniKP']
    prediction_script = PREDICTION_SCRIPTS['UniKP']
    job_dir = os.path.join(MEDIA_ROOT, 'jobs', str(public_id))
    input_temp_file = os.path.join(job_dir, f'input_{public_id}.csv')
    output_temp_file = os.path.join(job_dir, f'output_{public_id}.csv')

    total_molecules = len(sequences)
    job.total_molecules = total_molecules
    job.save(update_fields=["total_molecules"])

    valid_indices = []
    invalid_indices = []
    smiles_list = []
    valid_sequences = []
    alphabet = set('ACDEFGHIKLMNPQRSTVWY')
    # Process substrates and update progress
    for idx, (seq, substrate) in enumerate(zip(sequences, substrates)):
        mol = convert_to_mol(substrate)
        seq_valid = all(c in alphabet for c in seq)
        job.molecules_processed += 1
        if mol and seq_valid:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
            valid_sequences.append(seq)
            valid_indices.append(idx)
        else:
            print(f"Invalid substrate at row {idx + 1}: {substrate}")
            invalid_indices.append(idx)
            job.invalid_molecules += 1

        # Save job progress after each molecule
        job.save(update_fields=["molecules_processed", "invalid_molecules"])

    # Update total predictions
    job.total_predictions = len(valid_indices)
    job.save(update_fields=["total_predictions"])

    # Prepare DataFrame for valid entries
    if valid_indices:
        df_input = pd.DataFrame({
            'Substrate SMILES': smiles_list,
            'Protein Sequence': valid_sequences
        })
        df_input.to_csv(input_temp_file, index=False)
    else:
        df_input = pd.DataFrame()

    # Run the prediction script
    predictions = [None] * total_molecules  # Initialize with None
    if not df_input.empty:
        try:
            command = [python_path, prediction_script, input_temp_file, output_temp_file, kinetics_type]
            run_prediction_subprocess(command, job)

            # Read the output file
            df_output = pd.read_csv(output_temp_file)
            predicted_values = df_output['Predicted Value'].tolist()

            # Merge predictions back into the original order
            for idx_in_valid_list, pred in enumerate(predicted_values):
                idx = valid_indices[idx_in_valid_list]
                predictions[idx] = pred

        except Exception as e:
            print("An error occurred while running the Unikp subprocess:")
            print(e)
            # Clean up temporary files
            if os.path.exists(input_temp_file):
                os.remove(input_temp_file)
            if os.path.exists(output_temp_file):
                os.remove(output_temp_file)
            raise e

    # Clean up temporary files
    if os.path.exists(input_temp_file):
        os.remove(input_temp_file)
    if os.path.exists(output_temp_file):
        os.remove(output_temp_file)

    return predictions, invalid_indices
