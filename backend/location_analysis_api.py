# file: location_analysis_api.py

from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from collections import Counter

app = FastAPI(title="Restaurant Location Analysis API")

# Load dataset
df = pd.read_csv(r"D:\Cognify\RestoSense\data\zomato.csv", encoding="ISO-8859-1")

# Clean column name if BOM exists
df.rename(columns={"ï»¿User ID": "User ID"}, inplace=True)

# Ensure necessary columns exist
required_columns = ['Restaurant ID', 'City', 'Locality', 'Cuisines', 'Aggregate rating']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column {col} is missing from dataset")

@app.get("/city_stats")
def city_stats(city: str = Query(..., description="City name to analyze")):
    city_data = df[df['City'].str.lower() == city.lower()]
    
    if city_data.empty:
        raise HTTPException(status_code=404, detail="City not found")
    
    # Number of restaurants
    total_restaurants = city_data['Restaurant ID'].nunique()
    
    # Average rating
    avg_rating = round(city_data['Aggregate rating'].mean(), 2)
    
    # Top cuisines
    cuisines_list = city_data['Cuisines'].dropna().tolist()
    cuisine_counter = Counter()
    for cuisine in cuisines_list:
        # Handle multiple cuisines per restaurant
        for c in cuisine.split(','):
            cuisine_counter[c.strip()] += 1
    top_cuisines = cuisine_counter.most_common(5)
    
    # Top localities by average rating
    locality_stats = city_data.groupby('Locality')['Aggregate rating'].mean().sort_values(ascending=False).head(5)
    top_localities = [{"Locality": loc, "Avg Rating": round(rating, 2)} for loc, rating in locality_stats.items()]
    
    return {
        "City": city,
        "Total Restaurants": total_restaurants,
        "Average Rating": avg_rating,
        "Top Cuisines": top_cuisines,
        "Top Localities": top_localities
    }
