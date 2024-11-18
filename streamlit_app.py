import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# تحميل النموذج المدرب
model = joblib.load('path_to_your_model_file.joblib')

# قائمة الخيارات المختلفة
gender_options = ['Female', 'Male']
insurance_options = ['Medicare', 'UnitedHealthcare', 'Aetna', 'Cigna', 'Blue Cross']
admission_options = ['Elective', 'Emergency', 'Urgent']
medication_options = ['Aspirin', 'Lipitor', 'Penicillin', 'Paracetamol', 'Ibuprofen']

# ترميز القيم النصية في القوائم
encoder_gender = LabelEncoder().fit(gender_options)
encoder_insurance = LabelEncoder().fit(insurance_options)
encoder_admission = LabelEncoder().fit(admission_options)
encoder_medication = LabelEncoder().fit(medication_options)

# واجهة المستخدم في التطبيق
st.title("Healthcare Prediction App")

# إدخال البيانات من المستخدم
age = st.slider("Age", 0, 100, 25)
gender = st.selectbox("Gender", gender_options)
insurance_provider = st.selectbox("Insurance Provider", insurance_options)
admission_type = st.selectbox("Admission Type", admission_options)
medication = st.selectbox("Medication", medication_options)

# تحويل القيم المدخلة إلى قيم رقمية
gender_encoded = encoder_gender.transform([gender])[0]
insurance_encoded = encoder_insurance.transform([insurance_provider])[0]
admission_encoded = encoder_admission.transform([admission_type])[0]
medication_encoded = encoder_medication.transform([medication])[0]

# تحضير البيانات للاستخدام في النموذج
input_data = np.array([[age, gender_encoded, insurance_encoded, admission_encoded, medication_encoded]])

# تشغيل التنبؤ عند الضغط على الزر
if st.button("Predict Test Result"):
    prediction = model.predict(input_data)
    
    # تم تعريف نتائج التنبؤ (حسب ما هو منطقي مع نوع بياناتك)
    test_results = ['Inconclusive', 'Normal', 'Abnormal']  # أو اختر النتيجة المناسبة التي تتناسب مع فئات النموذج
    result = test_results[prediction[0]]
    
    # عرض النتيجة للمستخدم
    st.write(f"The predicted test result is: **{result}**")
