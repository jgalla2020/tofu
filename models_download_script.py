import os
from transformers import AutoModelForCausalLM, AutoTokenizer

username = os.getenv("USER")
hf_home = f"/state/partition1/user/{username}/cache/huggingface"
os.makedirs(hf_home, exist_ok=True)

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = hf_home

# Function to download models
def download_model(model_name, hf_key, token):
    print(f"Downloading model and tokenizer for {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(hf_key, token=token)
    tokenizer = AutoTokenizer.from_pretrained(hf_key, token=token)
    # Save to the specified HF_HOME location
    model.save_pretrained(f"{hf_home}/{model_name}")
    tokenizer.save_pretrained(f"{hf_home}/{model_name}_tokenizer")
    print(f"{model_name} downloaded successfully.")

# Dictionary of models to download with their Hugging Face keys
models_to_download = {
    "llama2-7b": "NousResearch/Llama-2-7b-chat-hf",
    "phi": "microsoft/phi-1_5",
    "stablelm": "stabilityai/stablelm-3b-4e1t",
    "pythia-1.4": "EleutherAI/pythia-1.4b-deduped",
}

# Get your Hugging Face token from an environment variable
hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

# Download each model
for model_name, hf_key in models_to_download.items():
    download_model(model_name, hf_key, hf_token)

print("Congratulations! The models have been downloaded!")
