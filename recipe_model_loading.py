from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

ingredients = "eggs, spinach, cheese"
prompt = f"Given these ingredients: {ingredients}. Suggest a recipe to cook."

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(recipe)