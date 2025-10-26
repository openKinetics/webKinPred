import os
from pathlib import Path
import sys

RESULTS_DIR = Path(os.environ.get("KINFORM_DATA"))
# Python bin paths
ESM_BIN = os.environ.get("KINFORM_ESM_PATH")
ESMC_BIN = os.environ.get("KINFORM_ESMC_PATH")
T5_BIN = os.environ.get("KINFORM_T5_PATH")
PSEQ2SITES_BIN = os.environ.get("KINFORM_PSEQ2SITES_PATH")
# Binding sites prediction paths
BS_PRED_PATH = Path(os.environ.get("KINFORM_MEDIA_PATH")) / "pseq2sites" / "binding_sites_all.tsv"
# Precomputed embeddings paths
COMPUTED_EMBEDDINGS_PATHS = {
    "esm2": [Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info/esm2_layer_26", Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info/esm2_layer_29"],
    "esmc": [Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info/esmc_layer_24", Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info/esmc_layer_32"],
    "t5"  : [Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info/prot_t5_layer_19", Path(os.environ.get("KINFORM_MEDIA_PATH")) / "sequence_info/prot_t5_last"],
}