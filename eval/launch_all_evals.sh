#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Set default model if not provided
MODEL_PATH=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
LOG_DIR=${2:-"None"}

echo "============================================"
echo "Starting evaluation pipeline for model: $MODEL_PATH"
echo "============================================"

if [ "$LOG_DIR" = "None" ]; then
    if [ -d "$MODEL_PATH" ]; then
        LOG_DIR="$MODEL_PATH"
    else
        LOG_DIR="$(pwd)/logs/${MODEL_PATH}-log"
        mkdir -p "$LOG_DIR"
    fi
else
    mkdir -p "$LOG_DIR"
fi
echo "Logs will be saved to: $LOG_DIR"

# Run LM Eval first
echo "Step 1: Running LM Evaluation Harness..."
accelerate launch run_lm_eval.py -m "$MODEL_PATH" --log_dir $LOG_DIR --tasks all

# Check if LM eval completed successfully
if [ $? -eq 0 ]; then
    echo "✅ LM Evaluation completed successfully"
else
    echo "❌ LM Evaluation failed with exit code $?"
    exit 1
fi

echo ""
echo "Step 2: Running AlpacaEval2 and Arena-Hard Evaluations..."
python run_arena_eval.py -m "$MODEL_PATH" --log_dir $LOG_DIR

# Check if eval completed successfully
if [ $? -eq 0 ]; then
    echo "✅ AlpacaEval2 and Arena-Hard Evaluations completed successfully"
else
    echo "❌ AlpacaEval2 and Arena-Hard Evaluations failed with exit code $?"
    exit 1
fi

echo ""
echo "Step 3: Running Safety Evaluations..."
cd safety-eval-fork/
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
python evaluation/run_all_generation_benchmarks.py --model_name_or_path "$MODEL_PATH" --log_dir "$LOG_DIR"

# Check if Safety eval completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Safety Evaluations completed successfully"
else
    echo "❌ Safety Evaluations failed with exit code $?"
    exit 1
fi

echo ""
echo "============================================"
echo "All evaluations completed for model: $MODEL_PATH"
echo "============================================"

# Display summary if it exists
if [ -f "logs/${MODEL_PATH}-log/summary.tsv" ]; then
    echo ""
    echo "Evaluation Summary:"
    echo "==================="
    cat "logs/${MODEL_PATH}-log/summary.tsv"
elif [ -f "$MODEL_PATH/summary.tsv" ]; then
    echo ""
    echo "Evaluation Summary:"
    echo "==================="
    cat "$MODEL_PATH/summary.tsv"
fi
