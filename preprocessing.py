import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna("Unknown")
    
    # Separate target and features
    X = df.drop("AggregateRating", axis=1)
    y = df["AggregateRating"]
    
    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical columns
    X = X.drop(categorical_cols, axis=1)
    X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    
    # Scale numeric features
    numeric_cols = X.select_dtypes(include=['float64','int64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Save encoder and scaler
    joblib.dump(encoder, "models/encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    return X, y
