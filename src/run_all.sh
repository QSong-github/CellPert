#!/bin/bash

TRAIN_DATA="/blue/qsong1/wang.qing/scPerturb/scUP/scGP/lincs/merged_all_965_with_morgan.h5ad"
TEST_DATA_DIR="/blue/qsong1/wang.qing/scPerturb/scUP/scGP/minitahoe"
MODEL_PATH="./best_gin_vae_model_node_level.pth"


if [ ! -f "${MODEL_PATH}" ]; then
    echo "train (10000 steps, batch_size=64)..."
    python main.py \
        --train_data_path "${TRAIN_DATA}" \
        --train_flag \
        --steps 10000 \
        --batch_size 64
    
    if [ $? -ne 0 ]; then
        echo "failed to train the model"
        exit 1
    fi
fi


echo "predicting 14个plates..."

for i in {1..14}; do
    echo ""
    echo "==================== Plate ${i} ===================="
    
    TEST_DATA="${TEST_DATA_DIR}/p${i}_with_morgan.h5ad"
    
    if [ ! -f "${TEST_DATA}" ]; then
        echo "skip ${TEST_DATA}"
        continue
    fi
    
    python main.py \
        --train_data_path "${TRAIN_DATA}" \
        --test_data_path "${TEST_DATA}" \
        --test_dataset_id ${i} \
        --test_flag \
        --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo "✓ Plate ${i} successfully predicted"
    else
        echo "✗ Plate ${i} failed to predict"
    fi
done

echo ""
echo "==================== done ===================="
echo "saved to:"
echo "  - output/all_plates_results.csv "
echo "  - output/predictions/plate_*_predictions.pkl "