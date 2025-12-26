from huggingface_hub import login, snapshot_download

login(token="your_token")

# hf_path = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
# save_path = (
#     "/checkpoint/memorization/apaulus/models/Llama-3.1-8B-Instruct-abliterated"
# )

# hf_path = "huihui-ai/Llama-3.2-1B-Instruct-abliterated"
# save_path = "/checkpoint/memorization/apaulus/models/Llama-3.2-1B-Instruct-abliterated"

# hf_path = "huihui-ai/Llama-3.3-70B-Instruct-abliterated"
# save_path = "/checkpoint/memorization/apaulus/models/Llama-3.3-70B-Instruct-abliterated"

# hf_path = "huihui-ai/Qwen3-8B-abliterated"
# save_path = "/checkpoint/memorization/apaulus/models/Qwen3-8B-abliterated"

# hf_path = "meta-llama/Llama-3.1-8B-Instruct"
# save_path = "/checkpoint/memorization/apaulus/models/Llama-3.1-8B-Instruct"

# hf_path = "meta-llama/Llama-3.2-1B-Instruct"
# save_path = "/checkpoint/memorization/apaulus/models/Llama-3.2-1B-Instruct"

hf_path = "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"
save_path = (
    "/checkpoint/memorization/apaulus/models/Huihui-Qwen3-4B-Instruct-2507-abliterated"
)

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
