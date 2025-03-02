import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("smartphone_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Smartphone Price Estimator")
st.write("Enter the smartphone specifications to predict the price.")

# User inputs
ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=4)
storage = st.number_input("Storage (GB)", min_value=16, max_value=1024, value=64)
screen_size = st.number_input("Screen Size (inches)", min_value=4.0, max_value=7.5, value=6.1)
camera_mp = st.number_input("Camera (MP)", min_value=2, max_value=108, value=12)
battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=4000)

# Predict button
if st.button("Predict Price"):
    features = np.array([[ram, storage, screen_size, camera_mp, battery]])
    scaled_features = scaler.transform(features)
    predicted_price = model.predict(scaled_features)
    st.success(f"Estimated Price: ${predicted_price[0]:,.2f}")
