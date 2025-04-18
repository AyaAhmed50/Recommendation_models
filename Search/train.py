from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle


df = pd.read_csv(r"Superstore-Data-1-review (1).csv",encoding='windows-1254')
df

# Assume df is already preprocessed and saved
product_names = df['Product Name'].dropna().unique()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(product_names)

# Save everything
with open("search_model.pkl", "wb") as f:
    pickle.dump((vectorizer, tfidf_matrix, product_names, df), f)
