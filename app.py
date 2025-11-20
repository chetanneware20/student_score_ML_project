import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load the trained model
# ---------------------------
MODEL_PATH = "Student_model (1).pkl"

st.title("📘 Student Performance Prediction App")
st.write("Provide the student details below to predict performance.")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------------
# Input fields (exact features used in training)
# -------------------------------------
st.subheader("Enter Student Data")

Assignments_Submitted = st.number_input("Assignments Submitted", min_value=0.0)
Attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
Hours_Studied = st.number_input("Hours Studied per Week", min_value=0.0)

# -------------------------------------
# Create input dataframe with exact feature order
# -------------------------------------
input_data = pd.DataFrame(
    {
        "Assignments_Submitted": [Assignments_Submitted],
        "Attendance": [Attendance],
        "Hours_Studied": [Hours_Studied],
    }
)

# FORCE the exact SAME ORDER used during training
input_data = input_data[
    ["Assignments_Submitted", "Attendance", "Hours_Studied"]
]

# -------------------------------------
# Prediction
# -------------------------------------
if st.button("Predict Score"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Score: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
