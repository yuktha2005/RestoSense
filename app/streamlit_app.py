import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    rating_model = joblib.load("models/rating_model.joblib")
    cuisine_model = joblib.load("models/cuisine_model.joblib")
    return rating_model, cuisine_model

rating_model, cuisine_model = load_models()

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    cols = ['Restaurant ID','Restaurant Name','City','Cuisines','Average Cost for two',
            'Currency','Aggregate rating']
    df = pd.read_csv(r"D:\Cognify\RestoSense\data\zomato.csv", usecols=cols, encoding="ISO-8859-1")
    df.rename(columns={"City": "Location", "Average Cost for two": "Budget"}, inplace=True)
    df["Cuisines"] = df["Cuisines"].astype(str)
    return df

df = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("RestoSense")
menu = ["Home", "Rating Prediction", "Cuisine Classification", "Restaurant Search", "Recommendations"]
choice = st.sidebar.radio("Explore", menu)

# -----------------------------
# Home Page
# -----------------------------
if choice == "Home":
    st.markdown("<h1 style='color:darkblue;text-align:center;'>Welcome to RestoSense 🍽️</h1>", unsafe_allow_html=True)
    st.write("Navigate using the sidebar to explore different features.")
    st.dataframe(df.head(10))

# -----------------------------
# Rating Prediction
# -----------------------------
elif choice == "Rating Prediction":
    st.header("Predict Restaurant Rating")

    # Ensure all cuisines and locations are strings
    cuisines_list = df["Cuisines"].dropna().astype(str).unique()
    location_list = df["Location"].dropna().astype(str).unique()

    # User selects multiple cuisines
    cuisines = st.multiselect("Select Cuisines", sorted(cuisines_list))
    location = st.selectbox("Select Location", sorted(location_list))
    
    # Budget slider
    budget = st.slider(
        "Budget (Average Cost for Two)",
        min_value=int(df["Budget"].min()),
        max_value=int(df["Budget"].max()),
        value=int(df["Budget"].mean())
    )

    if st.button("Predict Rating"):
        if not cuisines:
            st.error("Please select at least one cuisine")
        else:
            # Combine multiple cuisines as comma-separated string
            cuisines_input = ", ".join(cuisines)
            
            # Build input dataframe
            input_df = pd.DataFrame(
                [[cuisines_input, location, budget]],
                columns=["Cuisines", "Location", "Budget"]
            )

            # Ensure types match training
            input_df["Cuisines"] = input_df["Cuisines"].astype(str)
            input_df["Location"] = input_df["Location"].astype(str)
            input_df["Budget"] = pd.to_numeric(input_df["Budget"])

            # Predict
            rating = rating_model.predict(input_df)[0]
            st.markdown(f"<h2 style='color:green;'>Predicted Rating: {round(rating,2)} ⭐</h2>", unsafe_allow_html=True)

# -----------------------------
# Cuisine Classification
# -----------------------------
elif choice == "Cuisine Classification":
    st.header("Predict Restaurant Cuisine")

    desc = st.text_input("Enter restaurant description or features")

    if st.button("Predict Cuisine"):
        if not desc.strip():
            st.error("Please enter a description.")
        else:
            # Input MUST match model training column name
            input_df = pd.DataFrame({"Description": [desc]})

            cuisine = cuisine_model.predict(input_df)[0]

            st.markdown(
                f"<h2 style='color:purple;'>Predicted Cuisine: {cuisine}</h2>",
                unsafe_allow_html=True
            )

# -----------------------------
# Restaurant Search
# -----------------------------
elif choice == "Restaurant Search":
    st.header("Search Restaurant by Name or ID")
    search_term = st.text_input("Enter Restaurant Name or ID")
    if st.button("Search"):
        result = df[(df["Restaurant Name"].str.contains(search_term, case=False, na=False)) |
                    (df["Restaurant ID"].astype(str) == search_term)]
        if not result.empty:
            st.dataframe(result)
        else:
            st.error("No restaurant found.")

# -----------------------------
# Recommendations
# -----------------------------
elif choice == "Recommendations":
    st.header("Top Restaurant Recommendations")
    location = st.selectbox("Select Location", sorted(df["Location"].unique()))
    top_n = st.slider("Number of recommendations", 1, 10, 5)
    
    if st.button("Show Recommendations"):
        recommendations = df[df["Location"] == location].sort_values(by="Aggregate rating", ascending=False).head(top_n)
        
        for idx, row in recommendations.iterrows():
            # Set color based on rating
            if row['Aggregate rating'] >= 4.5:
                color = "#0a631f"  # green
            elif row['Aggregate rating'] >= 3.5:
                color = "#d7a913"  # yellow
            else:
                color = "#713F43"  # red
            
            st.markdown(f"""
                <div style="background-color:{color}; padding:15px; border-radius:10px; margin-bottom:10px;">
                <h3 style='margin:0'>{row['Restaurant Name']}</h3>
                <p style='margin:0'><b>Cuisines:</b> {row['Cuisines']}</p>
                <p style='margin:0'><b>Rating:</b> {row['Aggregate rating']} ⭐</p>
                <p style='margin:0'><b>Budget:</b> {row['Budget']} {row['Currency']}</p>
                </div>
            """, unsafe_allow_html=True)






