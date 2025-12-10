import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os

# Load dataset
df = pd.read_csv(r"D:\Cognify\RestoSense\data\zomato.csv", encoding="ISO-8859-1")

# Replace NaN values
df.fillna("", inplace=True)

# Build a combined text field for restaurants
df["combined_text"] = (
    df["Cuisines"].astype(str) + " " +
    df["Locality"].astype(str) + " " +
    df["City"].astype(str)
)

# TF-IDF for cuisines, locality, city
tfidf = TfidfVectorizer(stop_words="english")
text_features = tfidf.fit_transform(df["combined_text"])

# Numeric features
numeric_cols = ["Price range", "Average Cost for two", "Aggregate rating", "Votes"]
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[numeric_cols])

# Combine text + numeric
from scipy.sparse import hstack
final_features = hstack([text_features, numeric_features])

# Compute similarity matrix
similarity_matrix = cosine_similarity(final_features)

# Prepare save folder
os.makedirs("models", exist_ok=True)

# Save components
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump(scaler, "models/recommender_scaler.joblib")
joblib.dump(similarity_matrix, "models/similarity_matrix.joblib")
df.to_csv("models/restaurant_index.csv", index=False)

print("Recommendation engine model created successfully.")
