#!/bin/bash
# Train all 4 CatRange models sequentially (safe for single GPU)
set -e
cd /work/ssbio/aosinuga2/Python_work/CatRange
PYTHON=/work/ssbio/aosinuga2/envs/CatRange_env_gpu/bin/python
export PYTHONPATH=.

mkdir -p logs outputs

echo "=== TRAINING ALL 4 MODELS ===" | tee logs/train_all.log
echo "Started: $(date)" | tee -a logs/train_all.log

for config in kcat_esmc km_esmc kcat_esm2 km_esm2; do
    echo "" | tee -a logs/train_all.log
    echo ">>> Starting ${config} at $(date)" | tee -a logs/train_all.log
    $PYTHON scripts/cv_train.py \
        --config configs/${config}.yaml \
        --device cuda \
        2>&1 | tee logs/${config}_train.log
    echo ">>> Finished ${config} at $(date)" | tee -a logs/train_all.log
done

echo "" | tee -a logs/train_all.log
echo "=== ALL 4 MODELS COMPLETE ===" | tee -a logs/train_all.log
echo "Finished: $(date)" | tee -a logs/train_all.log
