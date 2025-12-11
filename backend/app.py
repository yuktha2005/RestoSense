from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
rating_model = joblib.load("models/rating_model.joblib")
encoder = joblib.load("models/encoder.joblib")
scaler = joblib.load("models/scaler.joblib")

@app.route("/predict-rating", methods=["POST"])
def predict_rating():
    data = request.json
    df_input = pd.DataFrame([data])
    
    # Encode categorical
    df_encoded = pd.DataFrame(encoder.transform(df_input), columns=encoder.get_feature_names_out(df_input.select_dtypes(include=['object']).columns))
    
    # Drop original categorical columns
    df_input = df_input.drop(df_input.select_dtypes(include=['object']).columns, axis=1)
    df_input = pd.concat([df_input.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
    
    # Scale numeric
    numeric_cols = df_input.select_dtypes(include=['float64','int64']).columns
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    
    # Predict
    pred = rating_model.predict(df_input)
    return jsonify({"predicted_rating": float(pred[0])})

if __name__ == "__main__":
    app.run(debug=True)
