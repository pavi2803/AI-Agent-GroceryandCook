from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load from your local model path
model_path = "local_models/phi3-mini-128k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32  # phi-3 uses float32 on MPS
).to("mps" if torch.backends.mps.is_available() else "cpu")

# Function to build prompt
def build_prompt(ingredients):
    messages = [
        {"role": "system", "content": "You are a helpful cooking assistant."},
        {"role": "user", "content": f"I have {ingredients}. What recipe can I make?"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Prepare prompt and inputs
prompt = build_prompt("onions, tomatoes, and eggs")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)