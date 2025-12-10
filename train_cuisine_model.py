import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv(r"D:\Cognify\RestoSense\data\zomato.csv", encoding="ISO-8859-1")

# Clean label: take only first cuisine
df["PrimaryCuisine"] = df["Cuisines"].apply(lambda x: str(x).split(",")[0].strip())

# Features to use
features = [
    "City", 
    "Locality",
    "Price range",
    "Average Cost for two",
    "Aggregate rating",
    "Votes",
    "Has Table booking",
    "Has Online delivery"
]

X = df[features]
y = df["PrimaryCuisine"]

# Identify categorical & numeric columns
categorical_cols = ["City", "Locality", "Has Table booking", "Has Online delivery"]
numeric_cols = ["Price range", "Average Cost for two", "Aggregate rating", "Votes"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Model pipeline
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200))
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print("Cuisine Classification Accuracy:", accuracy)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/cuisine_model.joblib")

print("Cuisine model saved successfully.")
