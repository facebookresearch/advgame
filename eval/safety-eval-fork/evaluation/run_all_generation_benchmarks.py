import json
import multiprocessing as mp
import os
from typing import Any
from dataclasses import dataclass
from collections import deque

import fire
import torch

from evaluation.utils import maybe_merge_lora_weights_and_save, upload_results_to_hf


def summary_results(output_log_path, log_dict):
    print()
    for key, value in log_dict.items(): print(key, ':', value)
    print()

    if not os.path.exists(output_log_path):
        with open(output_log_path, "w") as outfile: 
            outfile.write('\t'.join(log_dict.keys()) + '\n')

    with open(output_log_path, "a") as outfile: 
        outfile.write('\t'.join([str(x) for x in log_dict.values()]) + '\n')

@dataclass
class GPUTask:
    task_name: str
    gpu_ids: list[int]
    script_format: str

def run(task: GPUTask):
    gpu_ids_str = ",".join(map(str, task.gpu_ids))
    print(f"Running {task.task_name} on GPUs {gpu_ids_str}.")
    command = task.script_format.format(gpu_id=gpu_ids_str, task_name=task.task_name)
    os.system(command)


def allocate_gpus(available_gpus: deque[int], min_gpus_per_task: int) -> list[int]:
    """Allocate minimum required GPUs for a task."""
    if len(available_gpus) < min_gpus_per_task:
        return []
    
    allocated = []
    for _ in range(min_gpus_per_task):
        allocated.append(available_gpus.popleft())
    return allocated


def main(
        model_name_or_path: str,
        log_dir: str | None = None,
        model_input_template_path_or_name: str = "hf",
        min_gpus_per_task: int = 1,  # New parameter
        hf_revision: str = "main",
        filter_classifier_name: str | None = None,
        filter_model_override_path: str | None = None,
        task_timeout: int = 1800,  # Timeout in seconds (default: 30 minutes)
):
    if log_dir is None:
        if os.path.exists(model_name_or_path): log_dir = model_name_or_path
        else: log_dir = '../../logs/' + model_name_or_path + '-log'; os.makedirs(log_dir, exist_ok=True)
    log_dir_subdir = os.path.join(log_dir, 'safety_eval')
    os.makedirs(log_dir_subdir, exist_ok=True)
    report_output_path = os.path.join(log_dir_subdir, "metrics.json")
    save_individual_results_path = os.path.join(log_dir_subdir, "all.json")
    print(f"Log directory: {log_dir_subdir}")

    entrypoint_script_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "eval.py"
    )
    mp.set_start_method("spawn", force=True)

    if os.path.exists(model_name_or_path):
        model_name_or_path = os.path.abspath(model_name_or_path)

    temp_dir_for_merged_weights = maybe_merge_lora_weights_and_save(model_name_or_path)
    if temp_dir_for_merged_weights is not None:
        model_name_or_path = temp_dir_for_merged_weights

    task_names = [
        "harmbench:openai",
        "harmbench_precompute:openai",
        "xstest",
        "wildguardtest",
        "wildjailbreak:benign",
        "wildjailbreak:harmful",
        "do_anything_now",
        "or_bench:toxic",
    ]

    num_devices = 1 # torch.cuda.device_count()
    print(f"Number of devices: {num_devices}")
    
    if min_gpus_per_task > num_devices:
        raise ValueError(f"min_gpus_per_task ({min_gpus_per_task}) cannot be greater than available GPUs ({num_devices})")

    script_format = (
        "CUDA_VISIBLE_DEVICES={gpu_id} \\\n"
        f"python {entrypoint_script_path} generators \\\n"
        f"  --model_name_or_path {model_name_or_path} \\\n"
        f"  --model_input_template_path_or_name {model_input_template_path_or_name} --hf_revision {hf_revision}\\\n"
        "   --tasks {task_name} \\\n"
        f"  --report_output_path {report_output_path}" ".{task_name} \\\n"
        f"  --save_individual_results_path {save_individual_results_path}" ".{task_name} \\\n"
        "   --use_vllm \\\n"
        f"  --filter_classifier_name {filter_classifier_name} \\\n"
        f"  --filter_model_override_path {filter_model_override_path}"
    )

    with mp.Manager() as manager:
        # Initialize task queue and results
        tasks_queue = deque(task_names)
        results = {}
        
        while tasks_queue:
            current_tasks = []
            available_gpus = deque(range(num_devices))
            
            # Allocate GPUs to as many tasks as possible
            while tasks_queue and len(available_gpus) >= min_gpus_per_task:
                task_name = tasks_queue.popleft()
                gpu_ids = allocate_gpus(available_gpus, min_gpus_per_task)
                if gpu_ids:
                    current_tasks.append(GPUTask(
                        task_name=task_name,
                        gpu_ids=gpu_ids,
                        script_format=script_format
                    ))
                else:
                    tasks_queue.append(task_name)
                    break
            
            if not current_tasks:
                raise RuntimeError("Unable to allocate minimum required GPUs for any task")
            
            # Execute current batch of tasks
            with mp.Pool(processes=len(current_tasks)) as pool:
                res = []
                for task in current_tasks:
                    res.append(pool.apply_async(run, (task,)))
                
                # Wait for all processes in current batch to complete
                for task, r in zip(current_tasks, res):
                    try:
                        r.get(timeout=task_timeout)
                        results[task.task_name] = True
                    except mp.TimeoutError:
                        print(f"Timeout ({task_timeout}s) exceeded for {task.task_name}.")
                        results[task.task_name] = False
                    except Exception as e:
                        print(f"Error in {task.task_name}.")
                        print(e)
                        results[task.task_name] = False
                
                pool.close()
                pool.join()

    # Load and merge reports
    report_files = [report_output_path + f".{task_name}" for task_name in task_names]
    save_files = [save_individual_results_path + f".{task_name}" for task_name in task_names]

    all_report = {}
    for report_file in report_files:
        try:
            with open(report_file, "r") as f:
                report = json.load(f)
                all_report.update(report)
        except FileNotFoundError:
            print(f"Report file {report_file} not found.")
    
    summary_dict = {
        "harmbench_vanilla": all_report.get("harmbench:openai", {}).get("micro ASR (lower)", None),
        "harmbench_precompute": all_report.get("harmbench_precompute:openai", {}).get("micro ASR (lower)", None),
        "wildjailbreak:benign": all_report.get("wildjailbreak:benign", {}).get("macro ASR", None),
        "wildjailbreak:harmful": all_report.get("wildjailbreak:harmful", {}).get("macro ASR", None),
        "do_anything_now": all_report.get("do_anything_now", {}).get("macro ASR", None),
        "wildguardtest:vanilla": all_report.get("wildguardtest", {}).get("vanilla harm (lower)", None),
        "wildguardtest:adversarial": all_report.get("wildguardtest", {}).get("adversarial harm (lower)", None),
        "or_bench:toxic-refusal": all_report.get("or_bench:toxic", {}).get("refusal_rate_mean", None),
        "or_bench:toxic-harmful": all_report.get("or_bench:toxic", {}).get("harmfulness_rate_mean", None),
        "xstest:benign": all_report.get("xstest", {}).get("inverted_rta_all_safe", None),
        "xstest:harmful-refusal": all_report.get("xstest", {}).get("rta_all_contrast", None),
        "xstest:harmful-harmful": all_report.get("xstest", {}).get("rta_harmfulness_all_contrast", None),
    }

    if summary_dict["xstest:harmful-refusal"] is not None:
        summary_dict["xstest:harmful-refusal"] = 1 - summary_dict["xstest:harmful-refusal"]  # invert for use in leaderboard
    if summary_dict["or_bench:toxic-refusal"] is not None:
        summary_dict["or_bench:toxic-refusal"] = 1 - summary_dict["or_bench:toxic-refusal"]  # invert for use in leaderboard

    for key, value in summary_dict.items():
        summary_results(log_dir + '/summary.tsv', {
                'task': key, 
                'score': '%.2f' % (value * 100 if value is not None else -1) + '%',
            })

    with open(report_output_path, "w") as f:
        json.dump(all_report, f, indent=4)

    # Merge save files
    all_save = []
    for save_file in save_files:
        try:
            with open(save_file, "r") as f:
                save = json.load(f)
                all_save.extend(save)
        except FileNotFoundError:
            print(f"Save file {save_file} not found.")

    with open(save_individual_results_path, "w") as f:
        json.dump(all_save, f, indent=4)

    if temp_dir_for_merged_weights is not None:
        print(f"Removing temporary directory: {temp_dir_for_merged_weights}")
        os.system(f"rm -rf {temp_dir_for_merged_weights}")


if __name__ == "__main__":
    fire.Fire(main)
