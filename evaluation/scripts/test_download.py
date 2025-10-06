# download_safetensors_only.py
from huggingface_hub import snapshot_download

print("Downloading safetensors only...")
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    allow_patterns=["*.safetensors", "*.json"],
    ignore_patterns=["*.bin"],
)
print("Complete - check ~/.cache/huggingface/hub")