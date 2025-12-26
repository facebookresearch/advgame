#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Script to download Qwen2.5 models to /scratch/models on compute nodes
# Run with: srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash scripts/download_qwen_models.sh

set -e

MODELS_DIR="/scratch/models"

echo "=== Downloading Qwen models to ${MODELS_DIR} on $(hostname) ==="

# Create the models directory if it doesn't exist
mkdir -p "${MODELS_DIR}"

# Download Qwen2.5-32B-Instruct
echo "Downloading Qwen/Qwen2.5-32B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir "${MODELS_DIR}/Qwen2.5-32B-Instruct"

# Download Qwen2.5-7B-Instruct
echo "Downloading Qwen/Qwen2.5-7B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir "${MODELS_DIR}/Qwen2.5-7B-Instruct"

echo "=== Download complete on $(hostname) ==="
echo "Models saved to:"
echo "  - ${MODELS_DIR}/Qwen2.5-32B-Instruct"
echo "  - ${MODELS_DIR}/Qwen2.5-7B-Instruct"
