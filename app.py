
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="AI Medical Diagnosis System", layout="centered")

def set_background(image_file):
    import base64
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("images/background.png")

st.title("AI-Based Medical Diagnosis System")
diseases = {
    "Heart Disease": ["age", "sex", "cp", "trestbps", "chol"],
    "Diabetes": ["glucose", "bp", "skin_thickness", "insulin", "bmi"],
    "Pneumonia": ["fever", "cough", "chest_pain", "spo2", "age"],
    "Kidney Failure": ["bp", "sg", "al", "su", "bgr"],
    "Lung Disease": ["cough", "sob", "wheezing", "age", "smoking"]
}

disease = st.selectbox("Choose a disease to diagnose:", list(diseases.keys()))

model_path = f"models/{disease.lower().replace(' ', '_')}_model.pkl"
model = joblib.load(model_path)

st.subheader(f"Enter input values for {disease}:")
inputs = []
for feature in diseases[disease]:
    value = st.number_input(f"{feature}", step=0.1)
    inputs.append(value)

if st.button("Predict"):
    features = np.array(inputs).reshape(1, -1)
    prediction = model.predict(features)
    st.success(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
