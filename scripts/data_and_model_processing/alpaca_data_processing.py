from datasets import load_dataset, concatenate_datasets

seed = 42
hf_dir = "tatsu-lab/alpaca"
download_dir = "/checkpoint/memorization/apaulus/datasets/wildjailbreak_alpaca"

ds = load_dataset(hf_dir, "default")["train"]
ds = ds.remove_columns(["input", "output", "text"])
ds = ds.rename_column("instruction", "prompt")
ds = ds.add_column("category", ["benign"] * len(ds))
print(ds)


datasets = {}

num_test_examples = 4096
num_valid_examples = 2048

split_train_valid_test = ds.train_test_split(
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

datasets[f"alpaca_train_vanilla_benign"] = train
datasets[f"alpaca_valid_vanilla_benign_{len(valid)}"] = valid
datasets[f"alpaca_test_vanilla_benign_{len(test)}"] = test
datasets[f"alpaca_valid_vanilla_benign_{len(valid_64)}"] = valid_64
datasets[f"alpaca_valid_vanilla_benign_{len(valid_128)}"] = valid_128
datasets[f"alpaca_valid_vanilla_benign_{len(valid_256)}"] = valid_256
datasets[f"alpaca_valid_vanilla_benign_{len(valid_512)}"] = valid_512
datasets[f"alpaca_valid_vanilla_benign_{len(valid_1024)}"] = valid_1024
datasets[f"alpaca_valid_vanilla_benign_{len(valid_2048)}"] = valid_2048

for ds_name, ds_split in datasets.items():
    path = f"{download_dir}/{ds_name}/{ds_name}_data.jsonl"
    print(f"Saving {ds_name} to {path}")
    ds_split.to_json(path, orient="records", lines=True)
