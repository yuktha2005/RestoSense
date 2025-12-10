import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 1. Load dataset
df = pd.read_csv(r"D:\Cognify\RestoSense\data\Cuisine_rating.csv", encoding="ISO-8859-1")

# 2. Clean column names (remove BOM if exists)
df.rename(columns=lambda x: x.strip().replace("ï»¿", ""), inplace=True)

# 3. Choose features and target
features = ["Cuisines", "Location", "Budget"]
target = "Overall Rating"

# Ensure these columns exist
for col in features + [target]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataset. Available columns: {df.columns.tolist()}")

X = df[features]
y = df[target]

# 4. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Preprocessing
categorical_cols = ["Cuisines", "Location"]
numerical_cols = ["Budget"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# 6. Build model pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ]
)

# 7. Train model
model.fit(X_train, y_train)

# 8. Evaluate on test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.3f}")
print(f"Test R2: {r2:.3f}")

# 9. Ensure 'models' directory exists
os.makedirs("models", exist_ok=True)

# 10. Save complete model
joblib.dump(model, "models/rating_model.joblib")
print("Model saved successfully in 'models/rating_model.joblib'.")

