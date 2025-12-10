import pandas as pd
import joblib

# Load saved model
model = joblib.load("models/rating_model.joblib")

# Example input for prediction
data = pd.DataFrame([{
    "Cuisines": "Italian",
    "Location": "Mumbai",
    "Budget": 3
}])

# Predict rating
rating = model.predict(data)

print("Predicted Rating:", rating[0])
