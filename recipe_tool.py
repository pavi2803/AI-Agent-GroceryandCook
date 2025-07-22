from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer once
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_recipe(ingredients: str) -> str:
    prompt = f"Given these ingredients: {ingredients}. Suggest a recipe to cook."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Now, you can call get_recipe() many times without reloading the model
print(get_recipe("eggs, spinach, cheese"))
print(get_recipe("tomatoes, garlic, pasta"))
print(get_recipe("chicken, rice, curry powder"))
