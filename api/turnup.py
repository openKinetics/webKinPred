# turnup.py

import os
import subprocess
import pandas as pd
from rdkit import Chem
from api.utils.convert_to_mol import convert_to_mol
from api.models import Job  # Import the Job model to update progress
from webKinPred.settings import MEDIA_ROOT
try:
    from webKinPred.config_docker import PYTHON_PATHS, PREDICTION_SCRIPTS
except ImportError:
    try:
        from webKinPred.config_local import PYTHON_PATHS, PREDICTION_SCRIPTS
    except ImportError:
        PYTHON_PATHS = {}
        PREDICTION_SCRIPTS = {}

def turnup_predictions(sequences, substrates, products, public_id, protein_ids=None):
    """
    Run TurNup model on the given sequences, substrates, and products.

    Parameters:
    sequences: list of strings
        A list of protein sequences.
    substrates: list of strings
        A list of substrates; each element may contain multiple substrates separated by semicolons.
    products: list of strings
        A list of products; each element may contain multiple products separated by semicolons.
    public_id: int
        The job ID for tracking.
    protein_ids: list of strings, optional
        A list of protein accession numbers.

    Returns:
    list of floats
        A list of predicted kcat values. In the same order as the input sequences.
    """
    print("Running TurNup model...")

    # Get the Job object
    job = Job.objects.get(public_id=public_id)

    # Initialize progress fields
    job.molecules_processed = 0
    job.invalid_molecules = 0
    job.predictions_made = 0
    job.save(update_fields=["molecules_processed", "invalid_molecules", "predictions_made"])

    total_molecules = len(sequences)
    job.total_molecules = total_molecules
    job.save(update_fields=["total_molecules"])

    # Define paths
    python_path = PYTHON_PATHS['TurNup']
    prediction_script = PREDICTION_SCRIPTS['TurNup']
    job_dir = os.path.join(MEDIA_ROOT, 'jobs/'+str(public_id))

    input_temp_file = os.path.join(job_dir, f'input_{public_id}.csv')
    output_temp_file = os.path.join(job_dir, f'output_{public_id}.csv')

    # Set environment variables for the subprocess to use Docker-compatible paths
    env = os.environ.copy()
    try:
        from webKinPred.config_docker import DATA_PATHS
        env['TURNUP_MEDIA_PATH'] = DATA_PATHS['media']
        env['TURNUP_TOOLS_PATH'] = DATA_PATHS['tools']
    except (ImportError, KeyError):
        # If not using Docker config, don't set environment variables
        pass

    valid_indices = []
    invalid_indices = []
    subs_inchis = []
    prods_inchis = []
    valid_sequences = []
    alphabet = set('ACDEFGHIKLMNPQRSTVWY')
    predictions = [None] * total_molecules  # Initialize with None

    # Process reactions and update progress
    for idx, (seq, sub, prod) in enumerate(zip(sequences, substrates, products)):
        job.molecules_processed += 1

        sub_list = sub.split(';')
        prod_list = prod.split(';')
        sub_mols = [convert_to_mol(s.strip()) for s in sub_list]
        prod_mols = [convert_to_mol(p.strip()) for p in prod_list]
        seq_valid = all(c in alphabet for c in seq)
        # Check for invalid molecules
        if (None in sub_mols) or (None in prod_mols) or (not seq_valid):
            invalid_indices.append(idx)
            job.invalid_molecules += 1
        else:
            # Convert mols to InChIs
            sub_inchi = ';'.join(Chem.MolToInchi(mol) for mol in sub_mols)
            prod_inchi = ';'.join(Chem.MolToInchi(mol) for mol in prod_mols)

            subs_inchis.append(sub_inchi)
            prods_inchis.append(prod_inchi)
            valid_sequences.append(seq)
            valid_indices.append(idx)

        # Save job progress after each molecule
        job.save(update_fields=["molecules_processed", "invalid_molecules"])

    # Update total predictions
    job.total_predictions = len(valid_indices)
    job.save(update_fields=["total_predictions"])

    # Prepare DataFrame for valid entries
    if valid_indices:
        df_input = pd.DataFrame({
            'Substrates': subs_inchis,
            'Products': prods_inchis,
            'Protein Sequence': valid_sequences
        })
        df_input.to_csv(input_temp_file, index=False)
    else:
        df_input = pd.DataFrame()

    # Run the prediction script if we have valid data
    if not df_input.empty:
        try:
            result = subprocess.run(
                [python_path, prediction_script, input_temp_file, output_temp_file],
                check=True,
                capture_output=True,
                text=True,
                env=env  # Pass environment variables
            )
            print("Output:\n", result.stdout)
            print("Errors:\n", result.stderr)

            # Read the output file
            df_output = pd.read_csv(output_temp_file)
            predicted_values = df_output['kcat [s^(-1)]'].tolist()

            # Merge predictions back into the original order
            for idx_in_valid_list, pred in enumerate(predicted_values):
                idx = valid_indices[idx_in_valid_list]
                predictions[idx] = pred
                job.predictions_made += 1
                job.save(update_fields=["predictions_made"])

        except subprocess.CalledProcessError as e:
            print("An error occurred while running the TurNup subprocess:")
            print("Output:\n", e.stdout)
            print("Errors:\n", e.stderr)
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

    # Return predictions and invalid indices
    return predictions, invalid_indices
