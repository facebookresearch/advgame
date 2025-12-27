# AdvGame evals

## Install
Install conda if you don’t already have it:
https://www.anaconda.com/docs/getting-started/miniconda/install#linux

```
conda create -n advgame_eval python==3.12
conda activate advgame_eval
pip install -r requirements.txt
```

## Running Evaluations

Important NOTE: make sure tokenizer_config.json (stored under model_path) has a desired jinja chat_template. All generation-based evals pull chat template from it.

Important NOTE: we recommend running these evals on a node with >= 4 A100/H100 GPUs.

Our evaluations require OpenAI API set up (as a judge):
- To run alpaca-eval, you'll need to configure openai keys in: [data/openai_configs.yaml](data/openai_configs.yaml). Note: this assumes accessing OpenAI API via AzureOpenAI. More info and alternative options at [alpaca eval docs](https://github.com/tatsu-lab/alpaca_eval/tree/main/client_configs#configuring-openai).
- To run Arena-Hard, you'll need to configure openai keys in: [data/arena-hard-v0.1/configs.yaml](data/arena-hard-v0.1/configs.yaml). This assumes accessing OpenAI API via AzureOpenAI. To change this setting, modify configs.yaml and replace openai.AzureOpenAI inside [utils.py](utils.py).
- all safety evals (inside safety-eval-fork/) also rely on GPT-4o judge and it pull API info from [data/openai_configs.yaml](data/openai_configs.yaml). See more details [here](safety-eval-fork/src/classifier_models/openai_custom_classifier.py).

### Run all evaluations
```bash
bash launch_all_evals.sh YOUR_MODEL_PATH YOUR_LOG_DIR
```

Once evaluations are ready, summary.tsv will contain the resulting table of scores

### [Optional] Run individual evaluations
```bash
# Run lm-evaluation-harness (i.e. general knowledge and IFBench tasks)
# set '--tasks all' if you want to run all tasks
accelerate launch run_lm_eval.py -m YOUR_MODEL_PATH --tasks mmlu_llama ifbench --log_dir YOUR_LOG_DIR


# Run alpaca-eval and arena-hard (instruction following capability eval with gpt-4o as a judge)
python run_arena_eval.py -m YOUR_MODEL_PATH --log_dir YOUR_LOG_DIR

# Run safety evals
cd safety-eval-fork/
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
python evaluation/run_all_generation_benchmarks.py --model_name_or_path YOUR_MODEL_PATH --log_dir YOUR_LOG_DIR
```

Once evaluations are ready, summary.tsv will contain the resulting table of scores

