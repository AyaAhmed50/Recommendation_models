from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load TF-IDF model
with open("search_model.pkl", "rb") as f:
    vectorizer, tfidf_matrix, product_names, df = pickle.load(f)

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search_products(request: SearchRequest):
    query_vector = vectorizer.transform([request.query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    recommended_names = [product_names[i] for i in top_indices]
    
    recommended_df = df[df['Product Name'].isin(recommended_names)][['Product ID', 'Product Name', 'Rate']].drop_duplicates()
    
    return {
        "query": request.query,
        "recommendations": recommended_df.to_dict(orient="records")
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=1111)
