# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import re
import torch
import subprocess
import json
from copy import deepcopy
import concurrent.futures
from tqdm import tqdm
from utils import (
    jdump, 
    jload, 
    summary_results, 
    get_alpaca_eval_command,
    load_vllm_model, 
    form_llm_input, 
    generate_model_output_vllm,
    load_arena_questions,
    form_arena_llm_input,
    make_config,
    load_model_answers,
    JUDGE_SETTINGS,
    pairwise_judgment,
    print_leaderboard,
)

def extract_answer(text):
    """Extract text between <answer> and </answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def run_alpaca_eval(args, model, tokenizer, log_dir):

    data = jload("data/davinci_003_outputs.json")

    separate_input = True
    llm_input = form_llm_input(deepcopy(data), tokenizer.apply_chat_template, separate_input=separate_input)
    
    model_output = generate_model_output_vllm(llm_input, model, tokenizer)

    print(llm_input[0])
    print(model_output[0])

    output_log_file = log_dir + '/davinci_003_outputs_IH%d' % separate_input + '.json'

    for i in range(len(data)):
        data[i]['output'] = extract_answer(model_output[i].strip())
        data[i]['generator'] = args.model_name_or_path
        data[i]['instruction_only'] = data[i]['instruction']
        if data[i]['input'] != '': data[i]['instruction'] += '\n\n' + data[i]['input']
    
    jdump(data, output_log_file)

    alpaca_cmd = get_alpaca_eval_command("data/openai_configs.yaml", output_log_file)
    try: alpaca_log = subprocess.check_output(alpaca_cmd, shell=True, text=True)
    except subprocess.CalledProcessError: alpaca_log = 'None'
    found = False
    for item in [x for x in alpaca_log.split(' ') if x != '']:
        if args.model_name_or_path.split('/')[-1] in item: found = True; continue
        if found: begin_with = -1; in_response = float(item)/100; break # actually is alpaca_eval_win_rate
    if not found: begin_with = in_response = -1
    print(alpaca_log)

    summary_results(log_dir + '/summary.tsv', {
        'task': 'alpaca_eval2', 
        'score': '%.2f' % ((begin_with * 100) if begin_with > 0 else (in_response * 100)) + '%', 
    })


def judgement(question, answer, baseline_model, baseline, configs, args):
    output = {
        "uid": question["uid"],
        "category": question["category"],
        "judge": configs['judge_model']['model'],
        "model": args.model_name_or_path,
        "baseline": baseline_model,
        "games": []
    }

    # round 1
    result = pairwise_judgment(
        question=question,
        baseline=baseline,
        answer=answer,
        reference=None, # not used in our experiments
        configs=configs,
    )
    output["games"].append(result)
        
    # round 2
    result = pairwise_judgment(
        question=question,
        baseline=answer,
        answer=baseline,
        reference=None, # not used in our experiments
        configs=configs,
    )
    output["games"].append(result)

    with open(args.output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def run_arena_hard_eval(args, model, tokenizer, log_dir, version='v0.1'):
    question_file = f"data/arena-hard-{version}/question.jsonl"
    args.output_file = os.path.join(log_dir, f"arena_hard_{version}_results.jsonl")
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    questions = load_arena_questions(question_file)
    configs = make_config(f"data/arena-hard-{version}/configs.yaml")
    print("configs: ", configs)

    # step 1: prepare input for llm and generate answers
    llm_input = form_arena_llm_input(deepcopy(questions), tokenizer.apply_chat_template)
    model_output = generate_model_output_vllm(llm_input, model, tokenizer)

    print("llm_input[0]: ", llm_input[0])
    print("model_output[0]: ", model_output[0])

    model_answers = load_model_answers(f"data/arena-hard-{version}/model_answer")

    categories = set()

    # step 2: for each question, do pairwise judgment with baseline model
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = []
        for i, question in enumerate(questions):
            uid = question['uid']
            answer = extract_answer(model_output[i].strip())

            categories.add(question["category"])
            baseline_model = JUDGE_SETTINGS[question["category"]]["baseline"]
            baseline_msg = model_answers[baseline_model][uid]
            baseline = baseline_msg["messages"][-1]["content"]['answer']

            future = executor.submit(judgement, question, answer, baseline_model, baseline, configs, args)
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    # step 3: summarize results
    for category in categories:
        score = print_leaderboard([args.output_file], args.model_name_or_path, category)
        summary_results(log_dir + '/summary.tsv', {
            'task': f'arena-hard/{category}', 
            'score': score, 
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--skip_alpaca_eval', action='store_true')
    parser.add_argument('--skip_arena_hard_eval', action='store_true')
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()
    args.tensor_parallel_size = 1 # > 1 torch.cuda.device_count(), throws error for me
    # args.data_parallel_size = int(torch.cuda.device_count() / args.tensor_parallel_size)
    
    print(f"Running evaluation with args: {args}")

    log_dir = args.log_dir
    if log_dir is None:
        if os.path.exists(args.model_name_or_path): log_dir = args.model_name_or_path
        else: log_dir = 'logs/' + args.model_name_or_path + '-log'; os.makedirs(log_dir, exist_ok=True)

    # loading vllm model
    model, tokenizer = load_vllm_model(args.model_name_or_path, args.tensor_parallel_size)
    
    if not args.skip_alpaca_eval:
        run_alpaca_eval(args, model, tokenizer, log_dir)
    if not args.skip_arena_hard_eval:
        run_arena_hard_eval(args, model, tokenizer, log_dir)
