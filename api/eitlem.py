import os
import subprocess
import pandas as pd
from rdkit import Chem
from api.utils.convert_to_mol import convert_to_mol
from api.models import Job
from webKinPred.settings import MEDIA_ROOT
import numpy as np

try:
    from webKinPred.config_docker import PYTHON_PATHS, PREDICTION_SCRIPTS
except ImportError:
    try:
        from webKinPred.config_local import PYTHON_PATHS, PREDICTION_SCRIPTS
    except ImportError:
        PYTHON_PATHS = {}
        PREDICTION_SCRIPTS = {}


def run_prediction_subprocess(command, job, env=None):
    """
    Run a prediction subprocess and update job progress based on stdout.

    Parameters:
    - command: List of command-line arguments to run the subprocess.
    - job: Job object to update progress.
    - env: Environment variables to pass to subprocess.
    """
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,  # Pass environment variables
        )

        # Read stdout line by line
        for line in iter(process.stdout.readline, ""):
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
                        predictions_made, total_predictions = progress.split("/")
                        predictions_made = int(predictions_made)
                        total_predictions = int(total_predictions)
                        # Update the job object
                        job.predictions_made = predictions_made
                        job.total_predictions = total_predictions
                        job.save(
                            update_fields=["predictions_made", "total_predictions"]
                        )
                    except Exception as e:
                        print("Error parsing progress update:", e)
            else:
                # Handle other output if needed
                pass

        # Wait for the subprocess to finish
        process.wait()

        if process.returncode != 0:
            # An error occurred - check if it's memory-related
            if (
                process.returncode == -9 or process.returncode == 137
            ):  # SIGKILL (OOM killer)
                raise subprocess.CalledProcessError(
                    process.returncode, process.args, "Process killed by OOM killer"
                )
            else:
                raise subprocess.CalledProcessError(process.returncode, process.args)

    except Exception as e:
        print("An error occurred while running the subprocess:")
        print(e)
        raise e


def eitlem_predictions(
    sequences, substrates, public_id, protein_ids=None, kinetics_type="KCAT"
):
    print("Running EITLEM model...")

    # Get the Job object
    job = Job.objects.get(public_id=public_id)

    # Initialize progress fields
    job.molecules_processed = 0
    job.invalid_molecules = 0
    job.predictions_made = 0
    job.save(
        update_fields=["molecules_processed", "invalid_molecules", "predictions_made"]
    )

    # Define paths
    python_path = PYTHON_PATHS["EITLEM"]
    prediction_script = PREDICTION_SCRIPTS["EITLEM"]
    job_dir = os.path.join(MEDIA_ROOT, "jobs", str(public_id))
    input_temp_file = os.path.join(job_dir, f"input_{public_id}.csv")
    output_temp_file = os.path.join(job_dir, f"output_{public_id}.csv")
    # Set environment variables for the subprocess to use Docker-compatible paths
    env = os.environ.copy()
    try:
        from webKinPred.config_docker import DATA_PATHS

        env["EITLEM_MEDIA_PATH"] = DATA_PATHS["media"]
        env["EITLEM_TOOLS_PATH"] = DATA_PATHS["tools"]
    except (ImportError, KeyError):
        # If not using Docker config, don't set environment variables
        pass

    total_molecules = len(sequences)
    job.total_molecules = total_molecules
    job.save(update_fields=["total_molecules"])

    valid_indices = []
    invalid_indices = []
    smiles_list = []
    valid_sequences = []
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")

    # Process substrates and update progress
    for idx, (seq, substrate) in enumerate(zip(sequences, substrates)):
        mol = convert_to_mol(substrate)
        job.molecules_processed += 1
        seq_valid = all(c in alphabet for c in seq)
        if mol and seq_valid:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
            valid_sequences.append(seq)
            valid_indices.append(idx)
        else:
            invalid_indices.append(idx)
            job.invalid_molecules += 1
        # Save job progress after each molecule
        job.save(update_fields=["molecules_processed", "invalid_molecules"])

    # Update total predictions
    job.total_predictions = len(valid_indices)
    job.save(update_fields=["total_predictions"])

    # Prepare DataFrame for valid entries
    if valid_indices:
        df_input = pd.DataFrame(
            {"Substrate SMILES": smiles_list, "Protein Sequence": valid_sequences}
        )
        df_input.to_csv(input_temp_file, index=False)
    else:
        df_input = pd.DataFrame()

    # Run the prediction script
    predictions = [None] * total_molecules  # Initialize with None
    if not df_input.empty:
        try:
            command = [
                python_path,
                prediction_script,
                input_temp_file,
                output_temp_file,
                kinetics_type,
            ]
            run_prediction_subprocess(command, job, env)

            # Read the output file
            df_output = pd.read_csv(output_temp_file)
            predicted_values = df_output["Predicted Value"].tolist()

            # Merge predictions back into the original order
            for idx_in_valid_list, pred in enumerate(predicted_values):
                idx = valid_indices[idx_in_valid_list]
                if pred in ["None", "", np.nan, "nan"]:
                    predictions[idx] = None
                else:
                    predictions[idx] = pred

        except Exception as e:
            print("An error occurred while running the EITLEM subprocess:")
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
