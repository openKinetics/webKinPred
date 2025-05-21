import sys
import pandas as pd
import math
import torch
import esm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torch_geometric.data import Data, Batch
# Adjust the import paths according to your project structure
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
import hashlib
import os
import numpy as np
import json

ESM_EMB_DIR = "/home/saleh/webKinPred/media/sequence_info/esm1v"
SEQ2ID_PATH = "/home/saleh/webKinPred/media/sequence_info/seq_id_to_seq.json"


with open(SEQ2ID_PATH, "r") as f:
    seq_id_dict = json.load(f)

seq_to_id = {v: k for k, v in seq_id_dict.items()}
existing_ids = set(seq_id_dict.keys())

def generate_unique_seq_id(existing_ids, sequence):
    base_id = hashlib.sha256(sequence.encode()).hexdigest()[:12]
    suffix = 0
    new_id = base_id
    while new_id in existing_ids:
        suffix += 1
        new_id = f"{base_id}_{suffix}"
    return new_id

os.makedirs(ESM_EMB_DIR, exist_ok=True)
esm_model = None
alphabet = None
batch_converter = None

def load_esm_model_once():
    global esm_model, alphabet, batch_converter
    if esm_model is None:
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(
            model_location="/home/saleh/webKinPred/api/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt"
        )
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()

def main():
    if len(sys.argv) != 4:
        print("Usage: python eitlem_prediction_script.py input_file.csv output_file.csv kinetics_type")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    kinetics_type = sys.argv[3].upper()  # 'KCAT' or 'KM'

    # Load input data
    df_input = pd.read_csv(input_file)
    sequences = df_input['Protein Sequence'].tolist()
    substrates = df_input['Substrate SMILES'].tolist()

    # Define paths to model weights
    modelPath = {
        'KCAT':'/home/saleh/webKinPred/api/EITLEM/Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787',
        'KM': '/home/saleh/webKinPred/api/EITLEM/Weights/KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802',
        }

    if kinetics_type not in modelPath:
        print(f"Invalid kinetics type: {kinetics_type}")
        sys.exit(1)

    # Load EITLEM model
    if kinetics_type == 'KCAT':
        eitlem_model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
    elif kinetics_type == 'KM':
        eitlem_model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)

    eitlem_model.load_state_dict(torch.load(modelPath[kinetics_type], map_location=torch.device('cpu')))
    eitlem_model.eval()

    predictions = []
    total_predictions = len(sequences)
    i = 0  # Counter for predictions made

    for idx, (sequence, substrate) in enumerate(zip(sequences, substrates)):
        try:
            # Convert substrate SMILES to molecule
            mol = Chem.MolFromSmiles(substrate)
            if mol is None:
                raise ValueError(f"Invalid substrate SMILES: {substrate}")

            # Compute MACCS Keys
            mol_feature = MACCSkeys.GenMACCSKeys(mol).ToList()
            if sequence in seq_to_id:
                seq_id = seq_to_id[sequence]
            else:
                seq_id = generate_unique_seq_id(existing_ids, sequence)
                seq_id_dict[seq_id] = sequence
                seq_to_id[sequence] = seq_id
                existing_ids.add(seq_id)

            vec_path = os.path.join(ESM_EMB_DIR, f"{seq_id}.npy")
            if os.path.exists(vec_path):
                rep = np.load(vec_path)
            else:
                load_esm_model_once()
                if len(sequence) > 1023:
                    # take first 500 + last 500 residues
                    sequence = sequence[:500] + sequence[-500:]
                data = [("protein", sequence)]
                _, _, batch_tokens = batch_converter(data)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                tokens_len = batch_lens[0]
                rep = token_representations[0, 1:tokens_len - 1].cpu().numpy()
                np.save(vec_path, rep)
            # Use mean of per-residue embedding
            sequence_rep = torch.FloatTensor(rep)
            sample = Data(
                x=torch.FloatTensor(mol_feature).unsqueeze(0), 
                pro_emb=sequence_rep
            )

            input_data = Batch.from_data_list([sample], follow_batch=['pro_emb'])

            # Predict kinetics value
            with torch.no_grad():
                res = eitlem_model(input_data)
            prediction = math.pow(10, res[0].item())
            predictions.append(prediction)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            predictions.append(None)
        finally:
            i += 1
            print(f"Progress: {i}/{total_predictions} predictions made", flush=True)

    with open(SEQ2ID_PATH, "w") as f:
        json.dump(seq_id_dict, f, indent=2)

    # Save predictions to output file
    df_output = pd.DataFrame({
        'Substrate SMILES': substrates,
        'Protein Sequence': sequences,
        'Predicted Value': predictions
    })
    df_output.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
