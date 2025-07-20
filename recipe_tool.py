from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "local_models/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def suggest_dishes(ingredients):
    prompt = f"<bos><start_of_turn>user\nYou are a helpful cooking assistant. Given these ingredients: {ingredients}, suggest 3 dishes I can cook. Keep the response simple and focused on dish names.\n<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

out = suggest_dishes("eggs, tomatoes, onions, tortilla")
print(out)