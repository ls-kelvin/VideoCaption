# 下载 Qwen2-VL-7B-Instruct 模型到本地
# 运行前请确保安装了 huggingface_hub: pip install huggingface_hub

import os
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
LOCAL_DIR = "/data/flg/models/Qwen2-VL-7B-Instruct"

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    resume_download=True,
    # 如果下载速度慢，可以尝试使用镜像站点，设置环境变量 HF_ENDPOINT=https://hf-mirror.com
)

print("Download complete.")
