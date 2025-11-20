import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# Load the trained model
# ---------------------------
model_path = "Student_model (1).pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("📘 Student Score Prediction App")
st.write("Enter the student's details below to get the predicted score.")

# -------------------------------------
# Create input fields for user data
# (You can modify these based on your features)
# -------------------------------------
st.subheader("Enter Input Features")

# Example feature fields — adjust to match your model
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Create a dataframe for prediction
input_data = pd.DataFrame(
    {
        "Feature1": [feature1],
        "Feature2": [feature2],
        "Feature3": [feature3],
    }
)

# -------------------------------------
# Prediction
# -------------------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Score: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
