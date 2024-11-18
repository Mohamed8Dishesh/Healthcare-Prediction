pip install -r requirements.txt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load(r'D:\Unique_Interns\Proj2_Healthcare_Diagnosis_Prediction.joblib')

gender_options = ['Female', 'Male']
insurance_options = ['Medicare', 'UnitedHealthcare', 'Aetna', 'Cigna', 'Blue Cross']
admission_options = ['Elective', 'Emergency', 'Urgent']
medication_options = ['Aspirin', 'Lipitor', 'Penicillin', 'Paracetamol', 'Ibuprofen']



encoder_gender = LabelEncoder().fit(gender_options)
encoder_insurance = LabelEncoder().fit(insurance_options)
encoder_admission = LabelEncoder().fit(admission_options)
encoder_medication = LabelEncoder().fit(medication_options)



st.title("Healthcare Prediction App")



age = st.slider("Age", 0, 100, 25)
gender = st.selectbox("Gender", gender_options)
insurance_provider = st.selectbox("Insurance Provider", insurance_options)
admission_type = st.selectbox("Admission Type", admission_options)
medication = st.selectbox("Medication", medication_options)



gender_encoded = encoder_gender.transform([gender])[0]
insurance_encoded = encoder_insurance.transform([insurance_provider])[0]
admission_encoded = encoder_admission.transform([admission_type])[0]
medication_encoded = encoder_medication.transform([medication])[0]



input_data = np.array([[age, gender_encoded, insurance_encoded, admission_encoded, medication_encoded]])



if st.button("Predict Test Result"):
    prediction = model.predict(input_data)
    test_results = ['Inconclusive', 'Normal', 'Abnormal']
    result = test_results[prediction[0]]

    
    st.write(f"The predicted test result is: **{result}**")
