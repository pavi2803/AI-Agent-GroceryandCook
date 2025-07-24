from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer once
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_recipe(ingredients: str) -> str:
    prompt = f"What dishes can I make for dinner using only and just these ingredients: {ingredients}?"
    inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(**inputs, max_length=100)
    outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.3,
    top_p=0.8,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Now, you can call get_recipe() many times without reloading the model
# print(get_recipe("eggs, spinach, cheese, bread"))
print(get_recipe("bread, eggs, milk, butter, cinnamon powder, maple syrup"))
# print(get_recipe("cheese, bread, onions"))
