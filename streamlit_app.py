import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load(r'healthcare_model.joblib')

gender_options = ['Female', 'Male']
insurance_options = ['Medicare', 'UnitedHealthcare', 'Aetna', 'Cigna', 'Blue Cross']
admission_options = ['Elective', 'Emergency', 'Urgent']
medication_options = ['Aspirin', 'Lipitor', 'Penicillin', 'Paracetamol', 'Ibuprofen']

encoder_gender = LabelEncoder().fit(gender_options)
encoder_insurance = LabelEncoder().fit(insurance_options)
encoder_admission = LabelEncoder().fit(admission_options)
encoder_medication = LabelEncoder().fit(medication_options)

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: black;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #333;
    }
    .stButton>button {
        background-color: #444;
        color: white;
    }
    .stSlider>div>div>div {
        background-color: #444;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Healthcare Prediction App 🤖", page_icon="🤖", layout="centered")


st.title("Healthcare Prediction App")
st.markdown("### Predict the health test result based on the given information.")

age = st.slider("Age", 0, 100, 25, help="Select your age.")
gender = st.selectbox("Gender", gender_options, help="Select your gender.")
insurance_provider = st.selectbox("Insurance Provider", insurance_options, help="Select your insurance provider.")
admission_type = st.selectbox("Admission Type", admission_options, help="Select your admission type.")
medication = st.selectbox("Medication", medication_options, help="Select the medication you're taking.")

gender_encoded = encoder_gender.transform([gender])[0]
insurance_encoded = encoder_insurance.transform([insurance_provider])[0]
admission_encoded = encoder_admission.transform([admission_type])[0]
medication_encoded = encoder_medication.transform([medication])[0]

input_data = np.array([[age, gender_encoded, insurance_encoded, admission_encoded, medication_encoded]])

if st.button("Predict Test Result"):
    prediction = model.predict(input_data)
    test_results = ['Inconclusive', 'Normal', 'Abnormal']
    result = test_results[prediction[0]]

    st.markdown(f"### The predicted test result is: **{result}**", unsafe_allow_html=True)

    if result == 'Normal':
        st.markdown('<p style="color:green;">You are in normal condition.</p>', unsafe_allow_html=True)
    elif result == 'Abnormal':
        st.markdown('<p style="color:red;">There might be something abnormal. Consult with a doctor.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:orange;">The result is inconclusive. Further tests might be needed.</p>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("The test result is based on the input data. It is always recommended to consult a healthcare professional for a detailed diagnosis.")
