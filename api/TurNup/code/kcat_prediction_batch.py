import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import gc
from metabolite_preprocessing import *
from enzyme_representations import *
import sys
import pandas as pd
import os
from os.path import join

import warnings
warnings.filterwarnings("ignore")

# Use environment variables to determine paths
if os.environ.get('TURNUP_MEDIA_PATH'):
    # Docker environment
    data_dir = '/app/api/TurNup/data'
    SEQ_VEC_DIR = os.environ.get('TURNUP_MEDIA_PATH') + "/sequence_info/esm1b_turnup"
else:
    # Local environment
    data_dir = '/home/saleh/webKinPred/api/TurNup/data'
    SEQ_VEC_DIR = "/home/saleh/webKinPred/media/sequence_info/esm1b_turnup"

def kcat_prediction_batch(substrates, products, enzymes):
    """
    Process predictions one by one to avoid RAM issues.
    Load ESM1b model only if there are sequences that need embedding.
    """
    print("Step 1/3: Loading XGBoost model...")
    # Load XGBoost model once
    bst = pickle.load(open(join(data_dir, "saved_models", "xgboost", "xgboost_train_and_test.pkl"), "rb"))
    
    # Check if we need to load ESM1b model by doing a quick pass
    print("Step 2/3: Checking if ESM1b model is needed...")
    esm_model = None
    batch_converter = None
    esm_needed = False
    
    # Quick check: see if any sequences need embedding
    for enzyme in enzymes:
        enzyme_upper = enzyme.upper()
        df_enzyme_check = preprocess_enzymes([enzyme_upper])
        seqs = df_enzyme_check["model_input"].tolist()
        ids = resolve_seq_ids_via_cli(seqs)
        
        # Check if any sequences need embedding
        for seq_id in ids:
            vec_path = os.path.join(SEQ_VEC_DIR, f"{seq_id}.npy")
            if not os.path.exists(vec_path):
                esm_needed = True
                break
        
        if esm_needed:
            break
    
    # Load ESM1b model only if needed
    if esm_needed:
        print("ESM1b model needed - loading once for all predictions...")
        esm_model, batch_converter = load_esm1b_model()
    else:
        print("All sequences already cached - ESM1b model not needed!")
    
    predictions = []
    total_predictions = len(substrates)
    
    print("Step 3/3: Processing predictions one by one...")
    for i, (substrate, product, enzyme) in enumerate(zip(substrates, products, enzymes)):
        try:
            print(f"Progress: {i+1}/{total_predictions} predictions made", flush=True)
            
            # Process single reaction
            df_reaction = reaction_preprocessing(
                substrate_list=[substrate],
                product_list=[product]
            )
            
            # Process single enzyme using pre-loaded ESM model (if available)
            enzyme_upper = enzyme.upper()
            df_enzyme = calcualte_esm1b_ts_vectors(
                enzyme_list=[enzyme_upper],
                esm_model=esm_model,
                batch_converter=batch_converter
            )
            
            # Create single row DataFrame
            df_kcat = pd.DataFrame(data={
                "substrates": [substrate], 
                "products": [product],
                "enzyme": [enzyme_upper], 
                "index": [0]
            })
            
            # Merge reaction and enzyme data
            df_kcat = merging_reaction_and_enzyme_df(df_reaction, df_enzyme, df_kcat)
            df_kcat_valid = df_kcat.loc[df_kcat["complete"]]
            df_kcat_valid.reset_index(inplace=True, drop=True)
            
            if len(df_kcat_valid) > 0:
                # Calculate input matrix for single sample
                X = calculate_xgb_input_matrix(df=df_kcat_valid)
                dX = xgb.DMatrix(X)
                kcat = 10**bst.predict(dX)[0]  # Get first (and only) prediction
                predictions.append(kcat)
            else:
                predictions.append(None)  # Invalid sample
            
            # Clean up memory after each prediction
            del df_reaction, df_enzyme, df_kcat, df_kcat_valid
            if 'X' in locals():
                del X, dX
            gc.collect()
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            predictions.append(None)
    
    df_output = pd.DataFrame({
        "substrates": substrates,
        "products": products,
        "enzyme": enzymes,
        "kcat [s^(-1)]": predictions
    })
    df_output["complete"] = [p is not None for p in predictions]
    
    return df_output

def predict_kcat(X):
    bst = pickle.load(open(join(data_dir, "saved_models", "xgboost", "xgboost_train_and_test.pkl"), "rb"))
    dX = xgb.DMatrix(X)
    kcats = 10**bst.predict(dX)
    return(kcats)

def calculate_xgb_input_matrix(df):
    fingerprints = np.reshape(np.array(list(df["difference_fp"])), (-1,2048))
    ESM1b = np.reshape(np.array(list(df["enzyme rep"])), (-1,1280))
    X = np.concatenate([fingerprints, ESM1b], axis = 1)
    return(X)

def merging_reaction_and_enzyme_df(df_reaction, df_enzyme, df_kcat):
    df_kcat["difference_fp"], df_kcat["enzyme rep"] = "", ""
    df_kcat["complete"] = True

    for ind in df_kcat.index:
        diff_fp = list(df_reaction["difference_fp"].loc[df_reaction["substrates"] == df_kcat["substrates"][ind]].loc[df_reaction["products"] == df_kcat["products"][ind]])[0]
        esm1b_rep = list(df_enzyme["enzyme rep"].loc[df_enzyme["amino acid sequence"] == df_kcat["enzyme"][ind]])[0]

        if isinstance(diff_fp, str) and isinstance(esm1b_rep, str):
            df_kcat["complete"][ind] = False
        else:
            df_kcat["difference_fp"][ind] = diff_fp
            df_kcat["enzyme rep"][ind] = esm1b_rep
    return(df_kcat)

def main():
    if len(sys.argv) != 3:
        print("Usage: python kcat_prediction_script_batch.py input_file.csv output_file.csv")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read input data
    df_input = pd.read_csv(input_file)

    # Extract columns
    substrates = df_input['Substrates'].tolist()
    products = df_input['Products'].tolist()
    enzymes = df_input['Protein Sequence'].tolist()

    # Run predictions (batch processing)
    df_output = kcat_prediction_batch(substrates=substrates, products=products, enzymes=enzymes)
    df_output['Protein Sequence'] = df_output['enzyme']

    # Save output
    df_output.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
