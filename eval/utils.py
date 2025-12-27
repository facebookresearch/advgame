# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import io
import time
import random
import json
import transformers
import yaml
import pandas as pd
import openai
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from vllm import LLM, SamplingParams


def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    if isinstance(obj, (dict, list)): json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str): f.write(obj)
    else: raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r", num_samples=None):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    if num_samples is not None and num_samples > 0 and num_samples < len(jdict):
        random.seed(10)
        jdict = random.sample(jdict, num_samples)
        random.seed(time.time())
    return jdict


def summary_results(output_log_path, log_dict):
    print()
    for key, value in log_dict.items(): print(key, ':', value)
    print()

    if not os.path.exists(output_log_path):
        with open(output_log_path, "w") as outfile: 
            outfile.write('\t'.join(log_dict.keys()) + '\n')

    with open(output_log_path, "a") as outfile: 
        outfile.write('\t'.join([str(x) for x in log_dict.values()]) + '\n')


def get_alpaca_eval_command(openai_config_path, output_log_file):
    cmd = 'export OPENAI_CLIENT_CONFIG_PATH=%s\nalpaca_eval --is_overwrite_leaderboard' % openai_config_path
    
    if not os.path.exists(os.path.dirname(output_log_file)): 
        output_log_file = output_log_file.replace(os.path.dirname(output_log_file), os.path.dirname(output_log_file) + '-log')
    cmd += ' --model_outputs %s' % output_log_file
    if os.path.exists(output_log_file): print('Warning:', output_log_file, 'already exists and is going to be overwritten. Running utility on the existing file by')
    print('\n\n' + cmd + '\n\n')
    return cmd


def load_vllm_model(model_name_or_path, tensor_parallel_size=1):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path) # training code should the modified tokenizer to model_name_or_path
    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size, 
        trust_remote_code=True,
        max_model_len=16000,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def form_llm_input(data, apply_chat_template, separate_input=True):
    llm_input = []
    for i, d in enumerate(data): 
        d_item = deepcopy(d)

        if d['input'] != '': 
            if separate_input: 
                llm_input_i = apply_chat_template([{"role": "user",  "content": d_item['instruction']}, {"role": "input", "content": d_item['input']}], tokenize=False, add_generation_prompt=True)
            else: 
                llm_input_i = apply_chat_template([{"role": "user",  "content": d_item['instruction'] + '\n\n' + d_item['input']}], tokenize=False, add_generation_prompt=True)
        else: 
            llm_input_i = apply_chat_template([{"role": "user",  "content": d_item['instruction']}], tokenize=False, add_generation_prompt=True)
        
        llm_input.append(llm_input_i)
    return llm_input


def generate_model_output_vllm(llm_input, model, tokenizer):
    outputs = []
    sampling_params = SamplingParams(temperature=0, max_tokens=8192, stop=tokenizer.eos_token)
    for response in model.generate(llm_input, sampling_params): outputs.append(response.outputs[0].text)
    return outputs


#### utility functions for arena hard ####
###########################################


def load_arena_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def form_arena_llm_input(questions, apply_chat_template):
    llm_input = []
    for i, question in enumerate(questions): 
        llm_input_i = apply_chat_template([{"role": "user",  "content": question["prompt"]}], tokenize=False, add_generation_prompt=True)
        llm_input.append(llm_input_i)
    return llm_input


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[uid: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["uid"]] = line
        model_answers[model_name] = answer

    return model_answers


OG_ARENA_HARD_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."


JUDGE_SETTINGS = {
    "hard_prompt": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "coding": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "math": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "creative_writing": {
        "baseline": "gemini-2.0-flash-001",
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nWhen evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
    },
    "arena-hard-v0.1": {
        "baseline": "gpt-4-0314",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
}


# API setting constants
API_MAX_RETRY = 3
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = None


def get_score(judgment, patterns):
    import re
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def chat_completion_openai(
    messages: list, 
    temperature: float, 
    max_tokens: int, 
    api_dict=None, 
    **kwargs
):
    model = api_dict["model"]
    if api_dict:
        client = openai.AzureOpenAI(
            azure_endpoint=api_dict["azure_endpoint"],
            api_key=api_dict["api_key"],
            api_version=api_dict["api_version"],
        )
    else:
        client = openai.AzureOpenAI()
        
    if api_dict and "model_name" in api_dict:
        model = api_dict["model_name"]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = {
                "answer": completion.choices[0].message.content
            }
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


def pairwise_judgment(question, baseline, answer, reference, configs):
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": baseline,
        "ANSWER_B": answer,
    }
    
    if reference:
        prompt_args[f"REFERENCE"] = reference["messages"][-1]["content"]['answer']
        
    user_prompt = configs["prompt_template"].format(**prompt_args)
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[question["category"]]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    # build arguments for api completions
    kwargs = {
        "api_dict": configs["judge_model"],
        "messages": messages,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    output = chat_completion_openai(**kwargs)
    
    if output is None:
        return None

    score = get_score(output['answer'], configs["regex_patterns"])

    result = {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }
    return result


def load_judgments(judgement_files, weight=3):
    dfs = []
    for f_path in judgement_files:
        print(f"Loading {f_path} judgments...")
        dfs.extend([
            pd.read_json(f_path, lines=True)
        ])
    data = pd.concat(dfs).reset_index(drop=True)

    null_indices = data.games.map(lambda x: x[0] is None or x[1] is None or x[0]['score'] is None or x[1]['score'] is None)
    _data = data[~null_indices].reset_index(drop=True)
    
    print(f"Number of null judgments found: {len(data) - len(_data)}")
    
    # map label to score
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    _data['scores'] = _data.games.map(
        lambda x: label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
    )
    
    battles = _data[['uid', 'model', 'category', 'scores']].explode('scores').reset_index(drop=True)
    
    return battles

def format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline=None):
    leaderboard = pd.merge(
        mean_scores, 
        lower_scores, 
        on="model"
    ).merge(
        upper_scores, 
        on="model"
    )
    
    leaderboard["Scores (%)"] = leaderboard["scores"].map(lambda x: round(x * 100, 1))
    
    leaderboard["CI (%)"] = leaderboard.apply(
        lambda row: f"(-{round((row['scores'] - row['lower']) * 100, 1)} / +{round((row['upper'] - row['scores']) * 100, 1)})", 
        axis=1
    )
    
    _leaderboard = leaderboard.rename(
        columns={"model": "Model"}
    ).drop(
        columns=["lower", "upper", "scores"]
    )
    
    if baseline:
        _leaderboard = pd.concat(
            [_leaderboard, pd.DataFrame({"Model": baseline, "Scores (%)": 50.0, "CI (%)": "(-0.0 / +0.0)"}, index=[0])]
        )
    
    return _leaderboard.sort_values(by="Scores (%)", ascending=False).reset_index(drop=True)


def print_leaderboard(judgement_files, model_name, category):
    
    battles = load_judgments(judgement_files)
    
    baseline = JUDGE_SETTINGS[category]["baseline"]
    
    _battles = battles.drop(columns=['category'])[['model', 'scores']]
    
    # remove model path
    # _battles['model'] = _battles['model'].map(lambda x: x.split('/')[-1])
    
    bootstraps = pd.concat([
        _battles.groupby("model").sample(frac=1.0, replace=True).groupby("model").mean()
        for _ in tqdm(range(100))
    ])
    
    bootstraps["scores"] = bootstraps["scores"].astype(float)
    
    mean_scores = bootstraps.groupby("model").mean().reset_index()
    lower_scores = bootstraps.groupby("model").quantile(0.05).reset_index().rename(columns={"scores": "lower"})
    upper_scores = bootstraps.groupby("model").quantile(0.95).reset_index().rename(columns={"scores": "upper"})
    
    _leaderboard = format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline)
    
    print(f"##### Category: {category} #####")
    print(_leaderboard.to_string())

    model_score = _leaderboard[_leaderboard['Model'] == model_name]
    
    if not model_score.empty:
        return model_score['Scores (%)'].iloc[0]
    
    return None

###########################################