#!/usr/bin/zsh

RUN_PATH=$1

source "$HOME/.zshrc"

conda deactivate
conda activate fairseq2_2025-08-01 || conda activate fairseq2_2025-08-07

for STEP_TO_EVALUATE in "${@:2}"; do
    CHECKPOINT_NAME="checkpoint_step_$STEP_TO_EVALUATE"

    CHECKPOINT_DIR_DEF="$RUN_PATH/checkpoints"
    MODEL_PATH_DEF="$CHECKPOINT_DIR_DEF/step_$STEP_TO_EVALUATE/hg"
    LOG_DIR_DEF="$MODEL_PATH_DEF/eval"

    if [ ! -d "$MODEL_PATH_DEF" ]; then
        fairseq2 convert fs2_to_hg --checkpoint-dir $CHECKPOINT_DIR_DEF $CHECKPOINT_NAME $MODEL_PATH_DEF
    else
        echo "$MODEL_PATH_DEF exists already, skipping conversion."
    fi

    if [[ "$MODEL_PATH_DEF" == *"llama31"* ]]; then
        echo "Copying Llama-3.1-8B-Instruct config files..."
        cp -a configs/Llama-3.1-8B-Instruct/* $MODEL_PATH_DEF
    elif [[ "$MODEL_PATH_DEF" == *"qwen25"* ]]; then
        echo "Copying Qwen2.5-7B-Instruct config files..."
        cp -a configs/Qwen2.5-7B-Instruct/* $MODEL_PATH_DEF
    else
        echo "Model not recognized for config copy."
        exit 1
    fi
done

conda deactivate
conda activate advgame_eval

for STEP_TO_EVALUATE in "${@:2}"; do
    CHECKPOINT_NAME="checkpoint_step_$STEP_TO_EVALUATE"

    CHECKPOINT_DIR_DEF="$RUN_PATH/checkpoints"
    MODEL_PATH_DEF="$CHECKPOINT_DIR_DEF/step_$STEP_TO_EVALUATE/hg"
    LOG_DIR_DEF="$MODEL_PATH_DEF/eval"

    bash launch_all_evals.sh $MODEL_PATH_DEF $LOG_DIR_DEF
    python3 extract_scores.py $RUN_PATH
done

conda deactivate
