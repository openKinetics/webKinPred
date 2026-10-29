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
from pathlib import Path
from os.path import join

import warnings

warnings.filterwarnings("ignore")

# Use environment variables to determine paths
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_DEFAULT_MEDIA = _REPO_ROOT / "media"

_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from tools.gpu_embed_service.cache_io import resolve_missing_ids

_media_path = Path(os.environ.get("TURNUP_MEDIA_PATH", str(_DEFAULT_MEDIA)))
if os.environ.get("TURNUP_DATA_PATH"):
    data_dir = os.environ.get("TURNUP_DATA_PATH")
elif Path("/app/models/TurNup/data").exists():
    data_dir = "/app/models/TurNup/data"
else:
    data_dir = str((_HERE.parents[1] / "data").resolve())
SEQ_VEC_DIR = str((_media_path / "sequence_info" / "esm1b_turnup").resolve())
PREDICTION_BATCH_SIZE = 128


def kcat_prediction_batch(substrates, products, enzymes):
    """
    Process predictions in bounded chunks to avoid RAM issues.
    Load ESM1b model only if there are sequences that need embedding.
    """
    total_predictions = len(substrates)
    if total_predictions == 0:
        df_output = pd.DataFrame(
            {
                "substrates": substrates,
                "products": products,
                "enzyme": enzymes,
                "kcat [s^(-1)]": [],
            }
        )
        df_output["complete"] = []
        return df_output

    print("Step 1/4: Loading XGBoost model...")
    # Load XGBoost model once
    bst = pickle.load(
        open(
            join(data_dir, "saved_models", "xgboost", "xgboost_train_and_test.pkl"),
            "rb",
        )
    )

    # Check if we need to load ESM1b model by doing a quick pass
    print("Step 2/4: Checking if ESM1b model is needed...")
    esm_model = None
    batch_converter = None
    esm_needed = False

    enzyme_upper_by_row = [str(enzyme).upper() for enzyme in enzymes]
    check_df = preprocess_enzymes(enzyme_upper_by_row)
    check_ids = resolve_seq_ids_via_cli(check_df["model_input"].tolist())
    missing_ids, _ready_ids = resolve_missing_ids(
        check_ids,
        cache_dir=Path(SEQ_VEC_DIR).resolve(),
        suffix=".npy",
    )
    esm_needed = len(missing_ids) > 0

    # Load ESM1b model only if needed
    if esm_needed:
        print("ESM1b model needed - loading once for all predictions...")
        esm_model, batch_converter = load_esm1b_model()
    else:
        print("All sequences already cached - ESM1b model not needed!")

    predictions = [None] * total_predictions

    print("Step 3/4: Loading enzyme representations...")
    df_enzyme = calcualte_esm1b_ts_vectors(
        enzyme_list=list(dict.fromkeys(enzyme_upper_by_row)),
        esm_model=esm_model,
        batch_converter=batch_converter,
    )
    enzyme_rep_by_sequence = {
        str(sequence): rep
        for sequence, rep in zip(
            df_enzyme["amino acid sequence"], df_enzyme["enzyme rep"]
        )
    }

    # The ESM model is no longer needed after all sequence vectors are available.
    esm_model = None
    batch_converter = None
    df_enzyme = None
    gc.collect()

    print(
        f"Step 4/4: Processing predictions in batches of {PREDICTION_BATCH_SIZE}..."
    )
    for start in range(0, total_predictions, PREDICTION_BATCH_SIZE):
        end = min(start + PREDICTION_BATCH_SIZE, total_predictions)
        chunk_substrates = substrates[start:end]
        chunk_products = products[start:end]
        chunk_enzymes = enzyme_upper_by_row[start:end]
        df_reaction = None
        X = None
        dX = None
        fingerprints = None
        enzyme_reps = None

        try:
            df_reaction = reaction_preprocessing(
                substrate_list=chunk_substrates, product_list=chunk_products
            )

            valid_indices = []
            fingerprints = []
            enzyme_reps = []

            for local_idx, enzyme_upper in enumerate(chunk_enzymes):
                diff_fp = df_reaction["difference_fp"].iloc[local_idx]
                esm1b_rep = enzyme_rep_by_sequence.get(enzyme_upper)

                if isinstance(diff_fp, str) or isinstance(esm1b_rep, str):
                    continue
                if diff_fp is None or esm1b_rep is None:
                    continue

                diff_fp_array = np.asarray(diff_fp, dtype=np.float32)
                esm1b_array = np.asarray(esm1b_rep, dtype=np.float32)
                if diff_fp_array.shape != (2048,) or esm1b_array.shape != (1280,):
                    continue

                valid_indices.append(start + local_idx)
                fingerprints.append(diff_fp_array)
                enzyme_reps.append(esm1b_array)

            if valid_indices:
                X = np.concatenate(
                    [np.stack(fingerprints), np.stack(enzyme_reps)], axis=1
                ).astype(np.float32, copy=False)
                dX = xgb.DMatrix(X)
                kcats = 10 ** bst.predict(dX)
                for global_idx, kcat in zip(valid_indices, kcats):
                    predictions[global_idx] = float(kcat)

        except Exception as e:

            from tools.executor.policy import propagate_resource_error

            propagate_resource_error(e)
            print(f"Error processing batch {start}-{end}: {e}")
        finally:
            print(f"Progress: {end}/{total_predictions} predictions made", flush=True)
            df_reaction = None
            X = None
            dX = None
            fingerprints = None
            enzyme_reps = None
            gc.collect()

    df_output = pd.DataFrame(
        {
            "substrates": substrates,
            "products": products,
            "enzyme": enzymes,
            "kcat [s^(-1)]": predictions,
        }
    )
    df_output["complete"] = [p is not None for p in predictions]

    return df_output


def predict_kcat(X):
    bst = pickle.load(
        open(
            join(data_dir, "saved_models", "xgboost", "xgboost_train_and_test.pkl"),
            "rb",
        )
    )
    dX = xgb.DMatrix(X)
    from tools.executor.policy import limit_estimator
    bst = limit_estimator(bst)
    kcats = 10 ** bst.predict(dX)
    return kcats


def calculate_xgb_input_matrix(df):
    fingerprints = np.reshape(np.array(list(df["difference_fp"])), (-1, 2048))
    ESM1b = np.reshape(np.array(list(df["enzyme rep"])), (-1, 1280))
    X = np.concatenate([fingerprints, ESM1b], axis=1)
    return X


def merging_reaction_and_enzyme_df(df_reaction, df_enzyme, df_kcat):
    df_kcat["difference_fp"], df_kcat["enzyme rep"] = "", ""
    df_kcat["complete"] = True

    for ind in df_kcat.index:
        diff_fp = list(
            df_reaction["difference_fp"]
            .loc[df_reaction["substrates"] == df_kcat["substrates"][ind]]
            .loc[df_reaction["products"] == df_kcat["products"][ind]]
        )[0]
        esm1b_rep = list(
            df_enzyme["enzyme rep"].loc[
                df_enzyme["amino acid sequence"] == df_kcat["enzyme"][ind]
            ]
        )[0]

        if isinstance(diff_fp, str) and isinstance(esm1b_rep, str):
            df_kcat["complete"][ind] = False
        else:
            df_kcat["difference_fp"][ind] = diff_fp
            df_kcat["enzyme rep"][ind] = esm1b_rep
    return df_kcat


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python kcat_prediction_script_batch.py input_file.csv output_file.csv"
        )
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read input data
    df_input = pd.read_csv(input_file)

    # Extract columns
    substrates = df_input["Substrates"].tolist()
    products = df_input["Products"].tolist()
    enzymes = df_input["Protein Sequence"].tolist()

    # Run predictions (batch processing)
    df_output = kcat_prediction_batch(
        substrates=substrates, products=products, enzymes=enzymes
    )
    df_output["Protein Sequence"] = df_output["enzyme"]

    # Save output
    df_output.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
