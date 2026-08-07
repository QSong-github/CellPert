#!/bin/bash

# Train once if no checkpoint is present, then predict on all 14 Mini Tahoe plates.
# Usage, from the repository root:  bash src/run_all.sh
# Every path below is relative to the repository root; adjust DATA_DIR if the
# data is stored elsewhere.

# ==================== configuration ====================
DATA_DIR="./data"
TRAIN_DATA="${DATA_DIR}/lincs/merged_all_965_with_morgan.h5ad"
TEST_DATA_DIR="${DATA_DIR}/minitahoe"
MODEL_PATH="./src/best_gin_vae_model_node_level.pth"

# ==================== train if needed ====================
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Training the model..."
    python src/main.py \
        --train_data_path "${TRAIN_DATA}" \
        --train_flag \
        --epochs 5 \
        --batch_size 64

    if [ $? -ne 0 ]; then
        echo "Training failed."
        exit 1
    fi
fi

# ==================== predict ====================
echo "Predicting on the 14 plates..."

for i in {1..14}; do
    echo ""
    echo "==================== Plate ${i} ===================="

    TEST_DATA="${TEST_DATA_DIR}/p${i}_with_morgan.h5ad"

    if [ ! -f "${TEST_DATA}" ]; then
        echo "Skipped: ${TEST_DATA} not found"
        continue
    fi

    python src/main.py \
        --train_data_path "${TRAIN_DATA}" \
        --test_data_path "${TEST_DATA}" \
        --test_dataset_id ${i} \
        --test_flag \
        --batch_size 64

    if [ $? -eq 0 ]; then
        echo "Plate ${i} done"
    else
        echo "Plate ${i} failed"
    fi
done

echo ""
echo "==================== finished ===================="
echo "Results are written to:"
echo "  output/all_plates_results.csv          summary over plates"
echo "  output/predictions/plate_*_predictions.pkl   per-plate predictions"
