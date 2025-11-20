import streamlit as st
import pandas as pd
import pickle

MODEL_PATH = "Student_model (1).pkl"

st.title("📘 Student Performance Prediction App")
st.write("Provide the student details below to predict performance.")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.subheader("Enter Student Data")

# Add unique keys to avoid StreamlitDuplicateElementId
Assignments_Submitted = st.number_input(
    "Assignments Submitted", 
    min_value=0.0,
    key="assignments_key"
)

Attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    key="attendance_key"
)

Hours_Studied = st.number_input(
    "Hours Studied per Week",
    min_value=0.0,
    key="hours_key"
)

# Create dataframe
input_data = pd.DataFrame({
    "Assignments_Submitted": [Assignments_Submitted],
    "Attendance": [Attendance],
    "Hours_Studied": [Hours_Studied]
})

# 👉 Replace with your actual training order
input_data = input_data[
    ["Assignments_Submitted", "Attendance", "Hours_Studied"]
]

if st.button("Predict Score"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Score: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
