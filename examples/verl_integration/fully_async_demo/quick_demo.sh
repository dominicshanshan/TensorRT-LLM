#!/bin/bash

# This script is used to run the quick demo of fully_async with FSDP + TensorRT-LLM backend.

set -e

echo "================================================"
echo "VERL Fully Async Demo - FSDP + TensorRT-LLM"
echo "================================================"

# ============================================
# Step 1: Download Demo Dataset
# ============================================
DATASET_TYPE=${DATASET_TYPE:-"gsm8k"}  # Options: gsm8k, math, hh-rlhf
DATA_DIR="./demo_data/${DATASET_TYPE}"

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo ""
    echo "Step 1: Downloading ${DATASET_TYPE} dataset..."
    echo "----------------------------------------"
    bash download_demo_datasets.sh ${DATASET_TYPE}
else
    echo "Dataset already exists in ${DATA_DIR}"
fi

# ============================================
# Step 2: Set Configuration
# ============================================
echo ""
echo "Step 2: Setting up configuration..."
echo "----------------------------------------"

# Model selection (choose smaller model for demo)
MODEL_PATH=${MODEL_PATH:-"/scratch.trt_llm_data/llm-models/Qwen2-0.5B-Instruct"}  # Small model for quick demo

# Use small subset for quick demo
USE_SMALL_DATA=${USE_SMALL_DATA:-true}
if [ "$USE_SMALL_DATA" = true ]; then
    TRAIN_FILE="${DATA_DIR}/train_small.parquet"
    TEST_FILE="${DATA_DIR}/test_small.parquet"
    TOTAL_STEPS=$((64*10))  # Short training for small dataset
    echo "Using small dataset subset (1000 train, 100 test examples)"
else
    TRAIN_FILE="${DATA_DIR}/train.parquet"
    TEST_FILE="${DATA_DIR}/test.parquet"
    TOTAL_STEPS=$((256*50))  # Longer training for full dataset
    echo "Using full dataset"
fi