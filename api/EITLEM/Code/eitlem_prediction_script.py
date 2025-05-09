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

    # Load ESM1v model
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(
        model_location= "/home/saleh/webKinPred/api/EITLEM/Weights/esm1v/esm1v_t33_650M_UR90S_1.pt"
    )
    batch_converter = alphabet.get_batch_converter()
    model.eval()

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

            # Compute the MACCS Keys of substrate
            mol_feature = MACCSkeys.GenMACCSKeys(mol).ToList()

            # Extract protein representation
            data = [("protein", sequence)]
            _, _, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            sequence_representations = []
            for i_batch, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i_batch, 1 : tokens_len - 1])

            # Prepare input data for EITLEM model
            sample = Data(x=torch.FloatTensor(mol_feature).unsqueeze(0), pro_emb=sequence_representations[0])
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

    # Save predictions to output file
    df_output = pd.DataFrame({
        'Substrate SMILES': substrates,
        'Protein Sequence': sequences,
        'Predicted Value': predictions
    })
    df_output.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
