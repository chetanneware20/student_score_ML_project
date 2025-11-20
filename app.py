import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load the trained model
# ---------------------------
model_path = "Student_model (1).pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("📘 Student Score Prediction App")
st.write("Enter the student's details below to get the predicted score.")

# -------------------------------------
# Input fields matching the model's features
# -------------------------------------
Assignments_Submitted = st.number_input("Assignments Submitted", min_value=0.0)
Attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
Hours_Studied = st.number_input("Hours Studied", min_value=0.0)

# Make input dataframe with EXACT feature names
input_data = pd.DataFrame(
    {
        "Assignments_Submitted": [Assignments_Submitted],
        "Attendance": [Attendance],
        "Hours_Studied": [Hours_Studied],
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
