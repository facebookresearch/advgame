#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import csv
import os
import re
from glob import glob

# Fixed task order
TASKS = [
    "mmlu",
    "truthfulqa_mc1",
    "arc_challenge_llama",
    "meta_bbh",
    "ifeval",
    "ifbench",
    "alpaca_eval2",
    "arena-hard/arena-hard-v0.1",
    "harmbench_vanilla",
    "harmbench_precompute",
    "wildjailbreak:benign",
    "wildjailbreak:harmful",
    "do_anything_now",
    "wildguardtest:vanilla",
    "wildguardtest:adversarial",
    "or_bench:toxic-refusal",
    "or_bench:toxic-harmful",
    "xstest:benign",
    "xstest:harmful-refusal",
    "xstest:harmful-harmful",
]

SUMMARY_REL = os.path.join("checkpoints", "step_*", "hg", "eval", "summary.tsv")
STEP_RE = re.compile(r"step_(\d+)\b")


def parse_summary_tsv(path):
    """
    Parse a summary.tsv into dict: task -> numeric string (percent stripped).
    Ignores any tasks not in TASKS.
    """
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        # Skip header if present (robust to missing header)
        first = f.readline()

        # If the first line is not a header, process it as data
        def process_line(line):
            line = line.strip()
            if not line:
                return
            parts = line.split("\t")
            if len(parts) < 2:
                return
            task = parts[0].strip()
            raw = parts[1].strip()
            # only keep known tasks
            if task in TASKS:
                scores[task] = raw.replace("%", "").strip()

        if first:
            if "\t" in first and not first.lower().startswith("task\t"):
                process_line(first)  # it was data, not a header

        # Remaining lines
        for line in f:
            process_line(line)
    return scores


def main(model_path: str):
    # Find all candidate summary.tsv files
    pattern = os.path.join(model_path, SUMMARY_REL)
    candidates = glob(pattern)

    step_to_scores = {}

    for p in candidates:
        m = STEP_RE.search(p)
        if not m:
            continue
        step = int(m.group(1))
        if not os.path.isfile(p):
            continue
        try:
            scores = parse_summary_tsv(p)
        except Exception:
            # Skip unreadable/malformed files
            continue
        if scores:  # only keep steps with at least one known task
            step_to_scores[step] = scores

    if not step_to_scores:
        raise SystemExit(
            "No evaluated steps with summary.tsv found (containing known tasks)."
        )

    # Sort steps numerically
    steps_sorted = sorted(step_to_scores.keys())

    # Output path
    out_path = os.path.join(model_path, "scores_by_step.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write CSV: rows = steps, columns = TASKS (fixed order)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step"] + TASKS
        writer.writerow(header)
        for s in steps_sorted:
            row = [s]
            step_scores = step_to_scores[s]
            for task in TASKS:
                row.append(step_scores.get(task, ""))  # blank if missing
            writer.writerow(row)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Collect eval summaries into a transposed CSV by step (fixed task order)."
    )
    ap.add_argument(
        "model_path",
        help="Base model path containing checkpoints/step_*/hg/eval/summary.tsv",
    )
    args = ap.parse_args()
    main(args.model_path)
