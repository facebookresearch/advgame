# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from huggingface_hub import login, snapshot_download

login(token="your_token")

hf_path = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
save_path = (
    "DESTINATIN_PATH/Llama-3.1-8B-Instruct-abliterated"
)

# hf_path = "huihui-ai/Llama-3.2-1B-Instruct-abliterated"
# save_path = "DESTINATIN_PATH/Llama-3.2-1B-Instruct-abliterated"

# hf_path = "huihui-ai/Llama-3.3-70B-Instruct-abliterated"
# save_path = "DESTINATIN_PATH/Llama-3.3-70B-Instruct-abliterated"

# hf_path = "huihui-ai/Qwen3-8B-abliterated"
# save_path = "DESTINATIN_PATH/Qwen3-8B-abliterated"

# hf_path = "meta-llama/Llama-3.1-8B-Instruct"
# save_path = "DESTINATIN_PATH/Llama-3.1-8B-Instruct"

# hf_path = "meta-llama/Llama-3.2-1B-Instruct"
# save_path = "DESTINATIN_PATH/Llama-3.2-1B-Instruct"

# hf_path = "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"
# save_path = (
#     "DESTINATIN_PATH/Huihui-Qwen3-4B-Instruct-2507-abliterated"
# )

while True:
    try:
        snapshot_download(
            repo_id=hf_path,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        break
    except ConnectionError as e:
        print(f"ConnectionError: {e}. Retrying in 1s...")
        time.sleep(1)
