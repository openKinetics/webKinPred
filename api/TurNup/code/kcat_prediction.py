import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from metabolite_preprocessing import *
from enzyme_representations import *
import sys
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join

# Use environment variables to determine paths
if os.environ.get('TURNUP_MEDIA_PATH'):
    # Docker environment
    data_dir = '/app/api/TurNup/data'
else:
    # Local environment
    data_dir = '/home/saleh/webKinPred/api/TurNup/data'

def kcat_predicton(substrates, products, enzymes):
    #creating input matrices for all substrates:
    print("Step 1/3: Calculating numerical representations for all substrates and products.")
    df_reaction = reaction_preprocessing(substrate_list = substrates,
    								  product_list = products)

    print("Step 2/3: Calculating numerical representations for all enzymes.")
    enzymes = [enzyme.upper() for enzyme in enzymes]
    df_enzyme = calcualte_esm1b_ts_vectors(enzyme_list = enzymes)

    print("Step 3/3: Making predictions for kcat.")
    #Merging the reaction and the enzyme DataFrame:
    df_kcat = pd.DataFrame(data = {"substrates" : substrates, "products" : products,
    							 "enzyme" : enzymes, "index" : list(range(len(substrates)))})

    df_kcat = merging_reaction_and_enzyme_df(df_reaction, df_enzyme, df_kcat)
    df_kcat_valid, df_kcat_invalid = df_kcat.loc[df_kcat["complete"]], df_kcat.loc[~df_kcat["complete"]]
    df_kcat_valid.reset_index(inplace = True, drop = True)
    if len(df_kcat_valid) > 0:
        X = calculate_xgb_input_matrix(df = df_kcat_valid)
        kcats = predict_kcat(X)
        df_kcat_valid["kcat [s^(-1)]"] = kcats

    df_kcat = pd.concat([df_kcat_valid, df_kcat_invalid], ignore_index = True)
    df_kcat = df_kcat.sort_values(by = ["index"])
    df_kcat.drop(columns = ["index"], inplace = True)
    df_kcat.reset_index(inplace = True, drop = True)
    return(df_kcat)

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
        print("Usage: python kcat_prediction_script.py input_file.csv output_file.csv")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read input data
    df_input = pd.read_csv(input_file)

    # Extract columns
    substrates = df_input['Substrates'].tolist()
    products = df_input['Products'].tolist()
    enzymes = df_input['Protein Sequence'].tolist()

    # Run predictions
    df_output = kcat_predicton(substrates=substrates, products=products, enzymes=enzymes)
    df_output['Protein Sequence'] = df_output['enzyme']

    # Save output
    df_output.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()







