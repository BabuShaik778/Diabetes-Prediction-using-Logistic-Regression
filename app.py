import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered"
)

# Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Header
st.success("🩺 **Diabetes Prediction App**")
st.info("Machine Learning Health Check")
st.divider()

# Inputs in columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("👶 Pregnancies", min_value=0)
    glucose = st.number_input("🍬 Glucose", min_value=0)
    bp = st.number_input("💓 Blood Pressure", min_value=0)
    skin = st.number_input("🧪 Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("💉 Insulin", min_value=0)
    bmi = st.number_input("⚖️ BMI", min_value=0.0)
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("🎂 Age", min_value=1)

st.divider()

# Predict button
if st.button("🔍 Predict Result", use_container_width=True):
    data = np.array([[pregnancies, glucose, bp, skin,
                      insulin, bmi, dpf, age]])
    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.error("🔴 Result: **Diabetic (High Risk)**")
    else:
        st.success("🟢 Result: **Not Diabetic (Low Risk)**")

st.warning("⚠️ This prediction is based on a trained ML model and is for educational use only.")
