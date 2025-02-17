import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved models and preprocessor
pipeline = joblib.load("pipeline.pkl")
preprocessor = joblib.load("preprocessor.pkl")
feature_names = joblib.load("feature_names.pkl")  # Load feature names

# Streamlit UI
st.title("Real Estate Price Prediction")
st.write("Enter property details to predict the price")

# Predefined list of locations across India (expand this list as needed)
locations = [
    'Kanakapura Road', 'Indiranagar', 'Whitefield', 'Mumbai', 'Delhi', 'Bangalore',
    'Chennai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Goa',
    'Chandigarh', 'Lucknow', 'Surat', 'Nagpur', 'Vadodara', 'Bhopal', 'Patna', 'Coimbatore'
]

# User input fields
location = st.selectbox("Location", locations)
total_area = st.number_input("Total Area (sq ft)", min_value=500, max_value=5000, value=1200)
num_baths = st.slider("Number of Bathrooms", min_value=1, max_value=5, value=2)
balcony = st.selectbox("Balcony", ['Yes', 'No'])
description = st.text_input("Property Description", "2 BHK Apartment")
price_per_sqft = st.number_input("Price per sq ft", min_value=1000, max_value=20000, value=6000)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([[total_area, num_baths, price_per_sqft, location, description, balcony]],
                              columns=['Total_Area', 'Baths', 'Price_per_SQFT', 'Location', 'Description', 'Balcony'])

    # Preprocess input data
    input_data_preprocessed = preprocessor.transform(input_data)
    input_data_preprocessed_df = pd.DataFrame(input_data_preprocessed, columns=feature_names)

    # Make prediction
    prediction = pipeline.predict(input_data_preprocessed_df)[0]

    # Display the prediction result
    st.success(f"Predicted Price: ₹{prediction:,.2f}")

# Run by streamlit run index.py

