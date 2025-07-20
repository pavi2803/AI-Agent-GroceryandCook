from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-2b-it"
save_path = "local_models/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32
)

# Save locally
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
