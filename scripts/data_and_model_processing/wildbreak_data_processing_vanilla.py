# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datasets import load_dataset, concatenate_datasets

seed = 42
hf_dir = "allenai/wildjailbreak"
download_dir = "YOUR_DATA_SAVE_PATH"

ds = load_dataset(hf_dir, "train", delimiter="\t", keep_default_na=False)["train"]
ds = ds.remove_columns(["adversarial", "completion"])
print(ds)

ds = ds.rename_column("vanilla", "prompt")
ds = ds.rename_column("data_type", "category")
ds = ds.map(lambda x: {"category": x["category"].split("_")[1]})


datasets = {}

num_test_examples = 4096
num_valid_examples = 2048
num_train_examples = 2**16  # 65536


for category in ["benign", "harmful"]:
    ds_category = ds.filter(lambda x: x["category"] == category)

    split_train_valid_test = ds_category.train_test_split(
        test_size=num_test_examples + num_valid_examples, seed=seed, shuffle=True
    )
    train = split_train_valid_test["train"]
    valid_test = split_train_valid_test["test"]
    split_valid_test = valid_test.train_test_split(
        test_size=num_valid_examples, seed=seed, shuffle=True
    )
    valid = split_valid_test["train"]
    test = split_valid_test["test"]
    valid_64 = valid.select(range(64))
    valid_128 = valid.select(range(128))
    valid_256 = valid.select(range(256))
    valid_512 = valid.select(range(512))
    valid_1024 = valid.select(range(1024))
    valid_2048 = valid.select(range(2048))

    datasets[f"train_vanilla_{category}"] = train
    datasets[f"valid_vanilla_{category}_{len(valid)}"] = valid
    datasets[f"test_vanilla_{category}_{len(test)}"] = test
    datasets[f"valid_vanilla_{category}_{len(valid_64)}"] = valid_64
    datasets[f"valid_vanilla_{category}_{len(valid_128)}"] = valid_128
    datasets[f"valid_vanilla_{category}_{len(valid_256)}"] = valid_256
    datasets[f"valid_vanilla_{category}_{len(valid_512)}"] = valid_512
    datasets[f"valid_vanilla_{category}_{len(valid_1024)}"] = valid_1024
    datasets[f"valid_vanilla_{category}_{len(valid_2048)}"] = valid_2048

for ds_name, ds_split in datasets.items():
    path = f"{download_dir}/{ds_name}/{ds_name}_data.jsonl"
    print(f"Saving {ds_name} to {path}")
    ds_split.to_json(path, orient="records", lines=True)
