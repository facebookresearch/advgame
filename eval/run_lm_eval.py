# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import time
import os
import lm_eval
import torch
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.models.huggingface import HFLM
import nltk
nltk.download('punkt_tab')
from utils import (
    jdump, 
    jload, 
    summary_results, 
)

TASKS = ['mmlu', 'truthfulqa_mc1', 'arc_challenge_llama', 'meta_bbh', 'ifeval', 'ifbench']
TASKS_APPLY_CHAT_TEMPLATE = {
    'mmlu': False,  # likelihood tasks, so no chat template
    'truthfulqa_mc1': False,  # likelihood tasks, so no chat template
    'arc_challenge': False,  # likelihood task, so no chat template
    'bbh_cot_fewshot': False,  # generation task, but designed for base models so no chat_template is needed
    'ifeval': True,  # generation task
    'ifbench': True,  # generation task
    'mmlu_llama': True,  # generation task
    'arc_challenge_llama': True,  # generation task
    'meta_bbh': False,  # generation task, but designed for base models so no chat_template is needed
}

TASKS_FEWSHOT_AS_MULTITURN = {
    'mmlu_llama': True,
    'arc_challenge_llama': True,
}

def evaluate_and_log_task(task, lm_obj, task_manager, args, log_dir):

    log_file = f"{log_dir}/{task}-results.json"
    if os.path.exists(log_file) and args.skip_run_tasks:
        print(f"Skipping task {task} as results already exist in {log_file}")
        return
    time.sleep(5)

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=[task],
        fewshot_as_multiturn=TASKS_FEWSHOT_AS_MULTITURN.get(task, False),
        task_manager=task_manager,
        apply_chat_template=TASKS_APPLY_CHAT_TEMPLATE.get(task, False),
        batch_size=args.batch_size,
    )

    if results is not None:
        # print(f"Results for task {task}: {results["results"]}")
        if 'ifeval' in task or 'ifbench' in task:
            score = \
                (results["results"][task]['prompt_level_strict_acc,none'] +
                 results["results"][task]['inst_level_strict_acc,none'] +
                 results["results"][task]['prompt_level_loose_acc,none'] +
                 results["results"][task]['inst_level_loose_acc,none']) / 4
        elif task == 'bbh_cot_fewshot':
            score = results["results"][task]['exact_match,get-answer']
        elif 'acc,none' in results["results"][task]:
            score = results["results"][task]['acc,none']
        elif 'exact_match,strict_match' in results["results"][task]:
            score = results["results"][task]['exact_match,strict_match']
        else:
            score = results["results"][task]['exact_match,strict-match']

        summary_results(log_dir + '/summary.tsv', {
            'task': 'mmlu' if 'mmlu_llama' in task else task,
            'score': '%.2f' % (score * 100) + '%',
        })

        jdump(results, log_file)


def run_lm_eval(args):
    log_dir = args.log_dir
    if log_dir is None:
        if os.path.exists(args.model_name_or_path): log_dir = args.model_name_or_path
        else: log_dir = 'logs/' + args.model_name_or_path + '-log'; os.makedirs(log_dir, exist_ok=True)

    task_manager = lm_eval.tasks.TaskManager(include_path="llama_lm_eval_config")
    
    if 'all' in args.tasks:
        args.tasks = TASKS

    # is llama-3x? 
    if "llama" in args.model_name_or_path.lower() and "3" in args.model_name_or_path:
        args.tasks = [task if task != 'mmlu' else 'mmlu_llama' for task in args.tasks]

    lm_obj = None

    remaining_tasks = []
    for task in args.tasks:
        if 'alpaca_eval2' in task:
            continue  # handled separately
        # if TASKS_APPLY_CHAT_TEMPLATE.get(task, False):  # 'ifbench' in task or 'ifeval' in task:
        #     remaining_tasks.append(task)
        #     continue  # handled separately
        if lm_obj is None:
            lm_obj = HFLM(
                pretrained=args.model_name_or_path,
                batch_size=args.batch_size,
                max_length=8192,
            )
        evaluate_and_log_task(task, lm_obj, task_manager, args, log_dir)
    
    # del lm_obj
    # torch.cuda.empty_cache()
    # # use vllm for generation tasks
    # if len(remaining_tasks) > 0:
    #     lm_obj = VLLM(
    #         pretrained=args.model_name_or_path, 
    #         dtype="auto", 
    #         tensor_parallel_size=args.tensor_parallel_size, 
    #         # data_parallel_size=args.data_parallel_size,
    #         max_model_len=8192, 
    #         add_bos_token=False,
    #         batch_size=args.batch_size,
    #         gpu_memory_utilization=0.8,
    #     )
    #     for task in remaining_tasks:
    #         if 'ifbench' in task:
    #             task_manager = lm_eval.tasks.TaskManager(include_path="llama_lm_eval_config")
    #         evaluate_and_log_task(task, lm_obj, task_manager, args, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('-t', "--tasks", type=str, default=['all'], nargs='+')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--skip_run_tasks", action="store_true", default=False)
    args = parser.parse_args()
    args.tensor_parallel_size = torch.cuda.device_count()
    # args.data_parallel_size = int(torch.cuda.device_count() / args.tensor_parallel_size)
    
    print(f"Running evaluation with args: {args}")

    run_lm_eval(args)
