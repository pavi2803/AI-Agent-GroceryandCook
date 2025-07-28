import pandas as pd
import numpy as np

import pandas as pd

data = pd.read_csv("cleaned_recipe_data.csv")

data = data.dropna(subset=['Title', 'cleaned_ingredient_names'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['cleaned_ingredient_names'])

def search_similar_recipes(user_ingredients, top_k=3):
    user_vec = vectorizer.transform([user_ingredients])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    return data.iloc[top_indices]

def format_prompt(user_ingredients, retrieved_df):
    items = "\n".join(f"- {row['Title']}: {row['cleaned_ingredient_names']}" for _, row in retrieved_df.iterrows())
    return f"I have the following ingredients: {user_ingredients}.\nWhich of these dishes can I make?\n{items}\nJust name the best one."

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cpu")  # Force CPU usage

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model = model.to(device)

def get_suggestion(user_ingredients):
    matches = search_similar_recipes(user_ingredients)
    prompt = format_prompt(user_ingredients, matches)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



import psutil
print(f"Available memory: {psutil.virtual_memory().available / 1e9:.2f} GB")


print(get_suggestion("cream cheese, feta"))