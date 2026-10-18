# ========= START SCRIPT =========

#!/bin/bash

set -e

# MACE Polar fine-tuning script
# Run this from the folder containing mace_data/train.extxyz etc.

MODEL_NAME="polar_ft_1l_ca"
TRAIN_FILE="mace_data/train.extxyz"
VALID_FILE="mace_data/val.extxyz"
TEST_FILE="mace_data/test.extxyz"

# Basic checks
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: $TRAIN_FILE not found"
    exit 1
fi

if [ ! -f "$VALID_FILE" ]; then
    echo "ERROR: $VALID_FILE not found"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: $TEST_FILE not found"
    exit 1
fi

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

mace_run_train \
  --name="$MODEL_NAME" \
  --model="PolarMACE" \
  --foundation_model="polar-1-l" \
  --train_file="$TRAIN_FILE" \
  --valid_file="$VALID_FILE" \
  --test_file="$TEST_FILE" \
  --energy_key="REF_energy" \
  --forces_key="REF_forces" \
  --E0s="average" \
  --loss="weighted" \
  --energy_weight=1.0 \
  --forces_weight=10.0 \
  --r_max=4.0 \
  --lr=0.0005 \
  --weight_decay=5e-7 \
  --default_dtype="float64" \
  --device="cpu" \
  --batch_size=1 \
  --valid_batch_size=1 \
  --num_channels=64 \
  --restart_latest \
  --max_num_epochs=100 \
  --patience=15 \
  --seed=42

# ========= END SCRIPT =========
