from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load everything
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("collaborative_model.pkl", "rb") as f:
    model, trainset, df = pickle.load(f)

class RecommendRequest(BaseModel):
    customer_id: str  # original customer ID, not encoded

@app.post("/recommend")
def recommend_products(request: RecommendRequest):
    le_customer = label_encoders['Customer ID']
    le_product = label_encoders['Product ID']

    if request.customer_id not in le_customer.classes_:
        # New customer: popular products
        top_products = df.groupby('Product ID')['Sales'].sum().sort_values(ascending=False).head(5).index.tolist()
        product_names = le_product.inverse_transform(top_products)
        return {"customer_type": "new", "recommendations": list(product_names)}

    # Existing customer: personalized recommendations
    customer_encoded = le_customer.transform([request.customer_id])[0]
    customer_products = df[df['Customer ID'] == customer_encoded]['Product ID'].unique()

    recommendations = {
        pid: model.predict(customer_encoded, pid).est for pid in customer_products
    }

    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
    recommended_names = le_product.inverse_transform([r[0] for r in sorted_recommendations])
    ratings = [round(r[1], 2) for r in sorted_recommendations]

    return {
        "customer_type": "existing",
        "recommendations": [
            {"product": name, "estimated_rating": rating}
            for name, rating in zip(recommended_names, ratings)
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=1111)
