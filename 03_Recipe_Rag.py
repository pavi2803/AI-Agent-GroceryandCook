import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load and clean the recipe dataset
data = pd.read_csv("cleaned_recipe_data.csv")
data = data.dropna(subset=['Title', 'cleaned_ingredient_names'])

# Compute TF-IDF matrix for ingredient similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['cleaned_ingredient_names'])

# Search similar recipes using cosine similarity
def search_similar_recipes(user_ingredients, top_k=4):
    user_vec = vectorizer.transform([user_ingredients])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    return data.iloc[top_indices]

# Format the prompt for the LLM
def format_prompt(user_ingredients, retrieved_df):
    # Create readable descriptions of the retrieved recipes

    
    items = "\n".join(
        f"- {row['Title']}: ingredients include {', '.join(row['cleaned_ingredient_names'].split(',')[:6])}..."
        for _, row in retrieved_df.iterrows()
    )


    return (
        f"You are a creative chef. Here are some ingredients I have: {user_ingredients}.\n\n"
        f"Here are a few recipes that use the ingredients I have:\n"
        f"{items}\n\n"
        f"Based on all of these ingredients, suggest **one** delicious and creative dish I could make with all of my listed ingredients.\n"
        f"Only respond with the title of the dish"
    )

model_name = "google/flan-t5-base"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


import torch
from transformers import AutoModelForSeq2SeqLM

torch.backends.quantized.engine = 'qnnpack'  

# Apply dynamic quantization to reduce memory
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

device = torch.device("cpu")
model = model.to(device)

# Generate a recipe suggestion
def get_suggestion(user_ingredients):
    matches = search_similar_recipes(user_ingredients)
    prompt = format_prompt(user_ingredients, matches)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
    **inputs,
    do_sample=True,
    num_beams=3,
    temperature=0.2,
    max_new_tokens=25
)
    # outputs = model.generate(
    #     **inputs,
    #     do_sample=True,
    #     max_new_tokens=50,
    #     temperature=0.2,
    #     top_p=0.5,
    #     repetition_penalty=1.1,
    #     pad_token_id=tokenizer.eos_token_id,    
    #     eos_token_id=tokenizer.eos_token_id
    # )
    print("I think you should make 🍳 :")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

import psutil
print(f"Available memory: {psutil.virtual_memory().available / 1e9:.2f} GB")

# Run an example query
print(get_suggestion("peach pit, milk"))
