#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Script to download Llama3 models to /scratch/models on compute nodes
# Run with: srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash scripts/download_llama3_models.sh

set -e

MODELS_DIR="/scratch/models"

echo "=== Downloading Llama3 models to ${MODELS_DIR} on $(hostname) ==="

# Create the models directory if it doesn't exist
mkdir -p "${MODELS_DIR}"

# Download Meta-Llama-3.1-8B-Instruct-abliterated
echo "Downloading mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated..."
huggingface-cli download mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated --local-dir "${MODELS_DIR}/Meta-Llama-3.1-8B-Instruct-abliterated"

# Download Llama-3.1-8B-Instruct
echo "Downloading meta-llama/Llama-3.1-8B-Instruct..."
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir "${MODELS_DIR}/Llama-3.1-8B-Instruct"

# Download Llama-3.3-70B-Instruct-abliterated
echo "Downloading huihui-ai/Llama-3.3-70B-Instruct-abliterated..."
huggingface-cli download huihui-ai/Llama-3.3-70B-Instruct-abliterated --local-dir "${MODELS_DIR}/Llama-3.3-70B-Instruct-abliterated"

echo "=== Download complete on $(hostname) ==="
echo "Models saved to:"
echo "  - ${MODELS_DIR}/Meta-Llama-3.1-8B-Instruct-abliterated"
echo "  - ${MODELS_DIR}/Llama-3.1-8B-Instruct"
echo "  - ${MODELS_DIR}/Llama-3.3-70B-Instruct-abliterated"
