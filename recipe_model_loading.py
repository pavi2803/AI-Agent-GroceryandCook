from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Specify the model ID and local save path
model_id = "microsoft/phi-3-mini-128k-instruct"
save_path = "local_models/phi3-mini-128k-instruct"

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32  # Use float32 if you're on MPS
).to("mps")  # Or use "cpu" if needed

# Save to local directory
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
