from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi import Query


app = FastAPI()

# Load model once when API starts
model = joblib.load("models/rating_model.joblib")

@app.get("/")
def home():
    return {"message": "RestoSense API is running"}

@app.post("/predict-rating")
def predict_rating(cuisines: str, location: str, budget: float):
    input_data = pd.DataFrame([{
        "Cuisines": cuisines,
        "Location": location,
        "Budget": budget
    }])
    
    prediction = model.predict(input_data)[0]
    
    return {"predicted_rating": float(prediction)}
# Load model files
similarity_matrix = joblib.load("models/similarity_matrix.joblib")
restaurant_df = pd.read_csv("models/restaurant_index.csv")

def get_similar_restaurants(restaurant_id: int, top_n: int = 5):
    # Check ID exists
    if restaurant_id not in restaurant_df["Restaurant ID"].values:
        return {"error": "Restaurant ID not found"}

    index = restaurant_df.index[restaurant_df["Restaurant ID"] == restaurant_id].tolist()[0]

    # Get similarity scores
    scores = list(enumerate(similarity_matrix[index]))

    # Sort by similarity
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Top N similar restaurants (skip itself)
    similar_indices = [i for i, _ in sorted_scores[1 : top_n + 1]]

    return restaurant_df.iloc[similar_indices][
        ["Restaurant ID", "Restaurant Name", "Cuisines", "City", "Locality", "Aggregate rating"]
    ].to_dict(orient="records")
@app.get("/recommend")
def recommend_restaurants(restaurant_id: int = Query(...), top_n: int = 5):
    """
    Returns top-N similar restaurants based on cuisine, locality, price range, rating, and votes.
    """
    return get_similar_restaurants(restaurant_id, top_n)
