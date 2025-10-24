import os
import pandas as pd
import tempfile
import yaml
import pickle
from pathlib import Path
import subprocess
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ROOT, PSEQ2SITES_BIN

def _populate_initial(all_seq_ids, already_processed_ids, id_to_seq):
    missing_seq_ids = [sid for sid in all_seq_ids if sid not in already_processed_ids]

    print(f"{len(missing_seq_ids)} sequences missing binding-site predictions. \n Generating T5 features ...")
    success_dict = {seq_id: None for seq_id in all_seq_ids}
    reason_dict = {seq_id: None for seq_id in all_seq_ids}
    # populate process_list with (original_index, seq_id) tuples
    to_process = []
    for i, sid in enumerate(all_seq_ids):
        if sid not in missing_seq_ids:
            success_dict[sid] = True
            reason_dict[sid] = None
        elif len(id_to_seq[sid]) > 1499:
            success_dict[sid] = False
            reason_dict[sid] = "Sequence length exceeds 1499 residues."
        else:
            to_process.append((i, sid)) # append original index and seq_id
    return to_process, success_dict, reason_dict

def get_sites(
    seq_ids: list[str],
    id_to_seq: dict[str, str],
    binding_site_df: pd.DataFrame,
    batch_size: int = 4,
    save_path: str = "data/binding_sites/binding_site_cache.tsv",
    return_prot_t5: bool = False,
    ) -> tuple[dict[str, bool], dict[str, str], Path|None]:
    """
        Input:
            Protein sequence dictionary: {seq_id: sequence}
        Output:
            List of True and False (False if binding site failed)
            List of reasons for failure (None if successful)
    """
    # check which sequences are missing
    assert all(sid in id_to_seq for sid in seq_ids), "Some seq_ids not found in id_to_seq dict."
    already_processed_ids = binding_site_df['PDB'].unique().tolist()
    to_process, success_dict, reason_dict= _populate_initial(
        all_seq_ids=seq_ids,
        already_processed_ids=already_processed_ids,
        id_to_seq=id_to_seq
    )
    if len(to_process) == 0:
        print("All sequences already have binding-site predictions.")
        return success_dict, reason_dict, None
    
    features_output_path = ROOT / "results" / "pseq2sites_temp_features.pkl"
    # write tsv to tempfile (tsv with col 'PDB' and 'Sequence')
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tsv_path = Path(tmpdir) / "pseq2sites_input.tsv"
        # input_tsv_path = ROOT / "pseq2sites_input.tsv"
        with open(input_tsv_path, "w") as f:
            f.write("PDB\tSequence\n")
            for _, sid in to_process:
                f.write(f"{sid}\t{id_to_seq[sid]}\n")
        # generate features
        script_path = ROOT / "code" / "pseq2sites" / "Pseq2Sites" / "gen_features.py"
        gen_feat_cmd = [
            PSEQ2SITES_BIN, str(script_path),
            "--input", str(input_tsv_path),
            "--output", str(features_output_path),
            "--labels", "False"
        ]
        subprocess.run(gen_feat_cmd, check=True)
    # check which features failed and update success_list, reason_list, to_process
    with open(features_output_path, "rb") as f:
        features_output = pickle.load(f)
    IDs, seqs, feats = features_output
    for seq_id, seq, feat in zip(IDs, seqs, feats):
        if feat.shape[0] == 0:
            # skipped due to OOM
            success_dict[seq_id] = False
            reason_dict[seq_id] = "OOM error during T5 feature extraction."
        else:
            success_dict[seq_id] = True
            reason_dict[seq_id] = None
    # write temp configuration file 
    config_path = ROOT / "code" / "pseq2sites" / "Pseq2Sites" / "configuration_temp.yml"
    new_bspred_path = ROOT / "results" / "pseq2sites_temp_bspred"
    os.makedirs(new_bspred_path, exist_ok=True)
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    yaml_config["paths"]["prot_feats"] = str(features_output_path)
    yaml_config["paths"]["save_path"] = str(new_bspred_path)
    yaml_config["paths"]["result_path"] = str(new_bspred_path / "results.tsv")
    yaml_config["paths"]["model_path"] = str(ROOT / "code" / "pseq2sites" / "Pseq2Sites" / "results" / "model")
    yaml_config["train"]["batch_size"] = batch_size


    with open(config_path, "w") as f:
        yaml.dump(yaml_config, f)
    # run predictor
    prediction_script_path = ROOT / "code" / "pseq2sites" / "Pseq2Sites" / "test.py"
    predict_cmd = [
        PSEQ2SITES_BIN, str(prediction_script_path),
        "--config", str(config_path),
        "--labels", "False"
    ]
    subprocess.run(predict_cmd, check=True)
    # read results and update binding_site_df
    results_path = new_bspred_path / "results.tsv"
    results_df = pd.read_csv(results_path, sep="\t")
    binding_site_df = pd.concat([binding_site_df, results_df], ignore_index=True)
    # save updated cache
    binding_site_df.to_csv(save_path, sep="\t", index=False)
    # update and return success_list + reason_list
    for _, sid in to_process:
        if success_dict[sid] is None: #t5 did not fail
            binding_site_row = results_df[results_df['PDB'] == sid]
            if binding_site_row.empty:
                success_dict[sid] = False
                reason_dict[sid] = "Binding-site prediction failed."
            else:
                success_dict[sid] = True
                reason_dict[sid] = None
    if not return_prot_t5:
        # delete features file
        os.remove(features_output_path)
        features_output_path = None
    return success_dict, reason_dict, features_output_path