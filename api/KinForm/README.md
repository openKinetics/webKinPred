KinForm – Enzyme Kinetics Prediction
====================================
Code for the implementation and experiments of KinForm models for predicting enzyme–substrate kinetic parameters (kcat and KM) from protein sequence and SMILES.

If you just want to get predictions for your own data using our trained models, you can use the hosted web app (no setup required):
https://kineticxpredictor.humanmetabolism.org/

For details on the models and experiments, see the preprint:
https://arxiv.org/abs/2507.14639


Setup
-----
```bash
git clone https://github.com/Digital-Metabolic-Twin-Centre/KinForm.git
cd KinForm
pip install -r requirements.txt
```


Path A — Train and run with the Zenodo bundle
----------------------------------------------------------
Use this if you want to train/evaluate/predict with the same assets used in the paper.

### 1) Download the Zenodo bundle(s) and extract under `results/` (repo root)
Download <https://zenodo.org/records/17433514>. This includes:
- Trained models
- Protein embedddings
- Sequence ID mapping 
- pseq2sites binding-site predictions (for transparency/inspection; not required to run)

After extraction, your tree should include:

- `results/sequence_id_to_sequence.pkl`
- `results/protein_embeddings/...` (e.g., `esm2_layer_26/`, `esmc_layer_32/`, `prot_t5_last/`)
- `results/trained_models/...` (e.g `kcat_KinForm-L`, `kcat_KinForm-H`)
- Optional (for inspection only): `results/binding_sites/prediction.tsv`, `prediction_2.tsv` … `prediction_7.tsv` (Pseq2Sites outputs for all proteins)

### 2) Train models (run commands from `code/`)

```bash
cd code

# kcat
python main.py --mode train --task kcat --model_config UniKP
python main.py --mode train --task kcat --model_config KinForm-L
python main.py --mode train --task kcat --model_config KinForm-H

# KM
python main.py --mode train --task KM --model_config UniKP
python main.py --mode train --task KM --model_config KinForm-L
python main.py --mode train --task KM --model_config KinForm-H

# Optional: 5-fold KFold + GroupKFold evaluation (--train_test_split < 1.0 triggers cross-validation; 1.0 trains on all data)
python main.py --mode train --task kcat --model_config KinForm-L --train_test_split 0.8
```

### 3) Predict (predictions are saved in original/non-log units)

```bash
# Default dataset
python main.py --mode predict --task kcat --model_config KinForm-L --save_results ../predictions/kcat_L.csv

# Custom JSON
python main.py --mode predict --task KM --model_config UniKP --save_results ../predictions/km_unikp.csv --data_path ../my_km.json
```

Custom JSON format

- For kcat: array of {"sequence": str, "smiles": str, "value": float} (value is raw kcat, NOT log)
- For KM:   array of {"Sequence": str, "smiles": str, "log10_KM": float}


Path B — New proteins or full regeneration
-----------------------------------------
Use this if you want to train or predict using with new sequences, or rebuild all features locally. The pipeline generates embeddings and binding-site predictions then runs the training/prediction task.

### Prerequisites

#### 1) Initialize binding-site cache

Create or download `results/binding_sites/binding_sites_all.tsv`:

- **Option A (recommended):** Download the precomputed binding sites from [Zenodo](https://zenodo.org/records/17433514)
- **Option B:** Initialize an empty cache file:
  ```bash
  mkdir -p results/binding_sites
  echo -e "PDB\tPred_BS_Scores" > results/binding_sites/binding_sites_all.tsv
  ```

#### 2) Set up conda environments

The pipeline requires four separate Python environments for different embedding models. Create them as follows:

**ESM environment** (for ESM2 embeddings; Python 3.7):
```bash
conda create --name esm python=3.7 -y
conda activate esm
pip install torch fair-esm pandas tqdm
conda deactivate
```
See [ESM repository](https://github.com/facebookresearch/esm) for more details.

**ESMC environment** (for ESMC embeddings; Python 3.12):
```bash
conda create --name esmc python=3.12 -y
conda activate esmc
pip install esm pandas tqdm
conda deactivate
```
See [ESMC repository](https://github.com/evolutionaryscale/esm) for more details.

**ProtT5 environment** (for ProtT5 embeddings; Python 3.9):
```bash
conda create --name prott5 python=3.9 -y
conda activate prott5
pip install torch transformers sentencepiece pandas tqdm
conda deactivate
```
See [ProtTrans repository](https://github.com/agemagician/ProtTrans?tab=readme-ov-file) for more details.

**Pseq2Sites environment** (for binding-site prediction; Python 3.7):
```bash
conda create --name pseq2sites python=3.7 -y
conda activate pseq2sites
pip install torch transformers sentencepiece biopython==1.79 rdkit-pypi==2021.3.1 openbabel-wheel pandas tqdm
conda deactivate
```
See [Pseq2Sites repository](https://github.com/Blue1993/Pseq2Sites) for more details.

#### 3) Configure environment paths

Edit `code/config.py` to specify the Python binary paths for each environment. Locate each binary using:

```bash
conda activate <env_name>
which python
conda deactivate
```

Then update `config.py`:

```python
# Python bin paths
ESM_BIN = "/path/to/anaconda3/envs/esm/bin/python"
ESMC_BIN = "/path/to/anaconda3/envs/esmc/bin/python"
T5_BIN = "/path/to/anaconda3/envs/prott5/bin/python"
PSEQ2SITES_BIN = "/path/to/anaconda3/envs/pseq2sites/bin/python"
```

### Running the pipeline

Once the environments are configured, the pipeline handles all feature generation. 

```bash
cd code
# Train with custom data (see "Custom JSON format" above for data format)
python main.py --mode train --task kcat --model_config KinForm-L --data_path ../data/my_data.json

# Predict with custom data
python main.py --mode predict --task KM --model_config KinForm-H --save_results ../predictions/my_predictions.csv --data_path ../data/my_data.json
```

The pipeline will:
1. Check for existing embeddings in the cache
2. Generate missing protein embeddings using the appropriate environment (ESM, ESMC, ProtT5)
3. Generate binding-site predictions using Pseq2Sites (for weighted embeddings)
4. Cache computed features on disk ("results/protein_embeddings" for embeddings and "results/binding_sites/binding_sites_all.tsv" for binding site predictions.)
5. Proceed with training or prediction

**Note:** First-time runs may take significant time to compute embeddings for all proteins if no cuda/large dataset. Subsequent runs will use cached embeddings and complete much faster.


Acknowledgments
---------------
This work builds upon and benefits from several excellent open-source projects:

### Tools and Methods
- **[Pseq2Sites](https://github.com/Blue1993/Pseq2Sites)** – for generating binding site predictions
- **[UniKP](https://github.com/xxxx)** – baseline model for kinetic parameter prediction
- **[SMILES Transformer](https://github.com/DSPsleeporg/smiles-transformer)** – for generating small molecule embeddings

### Protein Embedding Models
- **[ESM (Facebook Research)](https://github.com/facebookresearch/esm)** – ESM2 protein language models
- **[ESM (Evolutionary Scale)](https://github.com/evolutionaryscale/esm)** – ESMC protein embeddings
- **[ProtTrans](https://github.com/agemagician/ProtTrans)** – ProtT5 protein embeddings

### Related Work
We also acknowledge the following projects that helped us understand the task of kinetic parameter prediction:
- **[DLKcat](https://github.com/SysBioChalmers/DLKcat)**
- **[TurNup](https://github.com/AlexanderKroll/kcat_prediction)**
- **[EITLEM-Kinetics](https://github.com/XvesS/EITLEM-Kinetics)**


