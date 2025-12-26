# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datasets import load_dataset, concatenate_datasets

seed = 42
hf_dir = "allenai/wildjailbreak"
download_dir = "/checkpoint/memorization/apaulus/datasets/wildjailbreak"

ds = load_dataset(hf_dir, "eval", delimiter="\t", keep_default_na=False)["train"]
print(ds)
ds = ds.remove_columns(["label"])
datasets = {}
ds = ds.rename_column("adversarial", "prompt")
ds = ds.rename_column("data_type", "category")
ds = ds.map(lambda x: {"category": x["category"].split("_")[1]})

for category in ["harmful", "benign"]:
    valid = ds.filter(lambda x: x["category"] == category)
    print(f"Category {category} has {len(valid)} examples")

    if category == "harmful":
        valid_64 = valid.select(range(64))
        valid_128 = valid.select(range(128))
        valid_256 = valid.select(range(256))
        valid_512 = valid.select(range(512))
        valid_1024 = valid.select(range(1024))
        valid_2000 = valid.select(range(2000))
        datasets[f"valid_adversarial_{category}_64"] = valid_64
        datasets[f"valid_adversarial_{category}_128"] = valid_128
        datasets[f"valid_adversarial_{category}_256"] = valid_256
        datasets[f"valid_adversarial_{category}_512"] = valid_512
        datasets[f"valid_adversarial_{category}_1024"] = valid_1024
        datasets[f"valid_adversarial_{category}_2000"] = valid_2000
    elif category == "benign":
        valid_64 = valid.select(range(64))
        valid_128 = valid.select(range(128))
        valid_210 = valid.select(range(210))
        datasets[f"valid_adversarial_{category}_64"] = valid_64
        datasets[f"valid_adversarial_{category}_128"] = valid_128
        datasets[f"valid_adversarial_{category}_210"] = valid_210
    else:
        raise ValueError(f"Unknown category {category}")


for ds_name, ds_split in datasets.items():
    path = f"{download_dir}/{ds_name}/{ds_name}_data.jsonl"
    print(f"Saving {ds_name} to {path}")
    ds_split.to_json(path, orient="records", lines=True)
