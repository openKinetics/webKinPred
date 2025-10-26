import os

# Base paths for Docker container
BASE_PATH = '/app'
DATA_PATH = '/app'

# Docker paths for conda environments (using the same names as exported)
PYTHON_PATHS = {
    'DLKcat': '/opt/conda/envs/dlkcat_env/bin/python',
    'EITLEM': '/opt/conda/envs/eitlem_env/bin/python', 
    'TurNup': '/opt/conda/envs/turnup_env/bin/python',
    'UniKP': '/opt/conda/envs/unikp/bin/python', 
    'KinForm': '/opt/conda/envs/kinform_env/bin/python',
    'esm2': '/opt/conda/envs/esm/bin/python',
    'esmc': '/opt/conda/envs/esmc/bin/python',
    't5': '/opt/conda/envs/prot_t5/bin/python',
    'pseq2sites': '/opt/conda/envs/pseq2sites/bin/python'
}

# Data paths for each model in Docker
DATA_PATHS = {
    'DLKcat': '/app/api/DLKcat/DeeplearningApproach/Data',
    'DLKcat_Results': '/app/api/DLKcat/DeeplearningApproach/Results',
    'EITLEM': '/app/api/EITLEM',
    'TurNup': '/app/api/TurNup/data',
    'UniKP': '/app/api/UniKP-main',
    "KinForm": '/app/api/KinForm/results',
    'media': '/app/media',
    'tools': '/app/tools',
}

# Prediction scripts paths (adapted for Docker container)
PREDICTION_SCRIPTS = {
    'DLKcat': '/app/api/DLKcat/DeeplearningApproach/Code/example/prediction_for_input.py',
    'EITLEM': '/app/api/EITLEM/Code/eitlem_prediction_script_batch.py',
    'TurNup': '/app/api/TurNup/code/kcat_prediction_batch.py',
    'UniKP': '/app/api/UniKP-main/run_unikp_batch.py',
    'KinForm': '/app/api/KinForm/code/main.py',
}

# Target databases (adapted for Docker container)
TARGET_DBS = {
    "DLKcat/UniKP": "/app/fastas/dbs/targetdb_dlkcat",
    "EITLEM/KinForm": "/app/fastas/dbs/targetdb_EITLEM", 
    "TurNup": "/app/fastas/dbs/targetdb_turnup"
}

# Other config variables
CONDA_PATH = "/opt/conda/bin/conda"
MODEL_LIMITS = {"EITLEM": 1024, "TurNup": 1024, "UniKP": 1000, "DLKcat": float("inf"), "KinForm-H": 1500, "KinForm-L": 1500}
SERVER_LIMIT = 10000
DEBUG = True
ALLOWED_FRONTEND_IPS = ["127.0.0.1", "localhost", "frontend", "backend"]

# Docker paths for experimental data
KM_CSV = '/app/media/experimental/km_experimental.csv'
KCAT_CSV = '/app/media/experimental/kcat_experimental.csv'