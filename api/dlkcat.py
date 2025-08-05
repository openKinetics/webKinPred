# dlkcat.py

import os
import subprocess
from rdkit import Chem
from api.utils.convert_to_mol import convert_to_mol
from api.models import Job  # Import the Job model to update progress
from webKinPred.config_local import PYTHON_PATHS, PREDICTION_SCRIPTS
from webKinPred.settings import MEDIA_ROOT

def dlkcat_predictions(sequences, substrates, public_id, protein_ids=None):
    """
    Run DLKCAT model on the given sequences and substrates.
    
    Parameters:
    sequences: list of strings
        A list of protein sequences.
    substrates: list of strings
        A list of substrate InChIs or SMILES.
    public_id: int
        The job ID for tracking purposes.
    protein_ids: list of strings, optional
        A list of protein accession numbers.

    Returns:
    list of floats
        A list of predicted kcat values. In the same order as the input sequences.
    """
    print("Running DLKCAT model...")

    # Get the Job object
    job = Job.objects.get(public_id=public_id)

    # Initialize progress fields
    job.molecules_processed = 0
    job.invalid_molecules = 0
    job.predictions_made = 0
    job.save()

    total_molecules = len(sequences)
    job.total_molecules = total_molecules
    job.save()

    # Define paths
    python_path = PYTHON_PATHS['DLKcat']
    prediction_script = PREDICTION_SCRIPTS['DLKcat']
    job_dir = os.path.join(MEDIA_ROOT, 'jobs/'+str(public_id))
    input_temp_file = os.path.join(job_dir, f'input_{public_id}.tsv')
    output_temp_file = os.path.join(job_dir, f'output_{public_id}.tsv')

    valid_indices = []
    invalid_indices = []
    smiles_list = []
    valid_sequences = []
    alphabet = set('ACDEFGHIKLMNPQRSTVWY')

    predictions = [None] * total_molecules  # Initialize with None

    # Process substrates and update progress
    for idx, (seq, substrate) in enumerate(zip(sequences, substrates)):
        job.molecules_processed += 1
        mol = convert_to_mol(substrate)
        seq_valid = all(c in alphabet for c in seq)
        if mol and seq_valid:
            mol = Chem.AddHs(mol)
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
            valid_sequences.append(seq)
            valid_indices.append(idx)
        else:
            print(f"Invalid substrate at row {idx + 1}: {substrate}")
            invalid_indices.append(idx)  # Rows are 1-indexed
            job.invalid_molecules += 1

        # Save job progress after each molecule
        job.save()

    # Update total predictions
    job.total_predictions = len(valid_indices)
    job.save()

    # Prepare input file for valid entries
    if valid_indices:
        with open(input_temp_file, 'w') as f:
            f.write('Substrate Name\tSubstrate SMILES\tProtein Sequence\n')
            for i in range(len(valid_indices)):
                name = 'noname'
                f.write(f'{name}\t{smiles_list[i]}\t{valid_sequences[i]}\n')

        # Run the prediction script using subprocess.Popen
        try:
            process = subprocess.Popen(
                [python_path, prediction_script, input_temp_file, output_temp_file],
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
                    # e.g., line = "Progress: 5/10 predictions made"
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

            # Read the output file
            predicted_values = []
            with open(output_temp_file, 'r') as f:
                next(f)  # Skip the header
                for line in f:
                    _, _, _, kcat_value = line.strip().split('\t')
                    try:
                        kcat_value = float(kcat_value)
                    except ValueError:
                        kcat_value = None
                    predicted_values.append(kcat_value)

            # Merge predictions back into the original order
            for idx_in_valid_list, pred in enumerate(predicted_values):
                idx = valid_indices[idx_in_valid_list]
                predictions[idx] = pred

        except Exception as e:
            print("An error occurred while running the DLKCAT subprocess:")
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

    # Return predictions and invalid indices
    return predictions, invalid_indices
