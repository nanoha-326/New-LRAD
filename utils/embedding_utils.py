import numpy as np
import pandas as pd
import pickle
from openai import OpenAI
import os

def get_openai_client(api_key):
    return OpenAI(api_key=api_key)

def get_embedding(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    res = client.embeddings.create(input=[text], model=model)
    return res.data[0].embedding

def save_embeddings(embeddings, path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_and_cache_faq_embeddings(faq_df, client, embed_path):
    if os.path.exists(embed_path):
        return load_embeddings(embed_path)
    embeddings = [get_embedding(q, client) for q in faq_df["質問"]]
    save_embeddings(embeddings, embed_path)
    return embeddings

