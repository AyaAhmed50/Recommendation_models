# train_model.py

import pandas as pd
from gensim.models import Word2Vec
import pickle

# Load and preprocess dataset
df = pd.read_csv("Superstore-Data-1-review (1).csv", encoding='windows-1254')
df['Full Description'] = df['Product Name'] + ' ' + df['Sub-Category']
df['Tokens'] = df['Full Description'].apply(lambda x: x.lower().split())

# Train Word2Vec model
model = Word2Vec(sentences=df['Tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Save model and dataframe
model.save("word2vec.model")
with open("product_data.pkl", "wb") as f:
    pickle.dump(df, f)

print("Training completed and model/data saved.")
