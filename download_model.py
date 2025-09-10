from huggingface_hub import hf_hub_download

repo_id = "SandLogicTechnologies/MedGemma-4B-IT-GGUF"
filename = "medgemma-4b-it-Q4_K_M.gguf"   # <- exact filename from HF repo

print("Downloading... this may take a while.")
path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
print("Downloaded to:", path)
