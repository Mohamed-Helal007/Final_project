import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("stroke_xgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Stroke Prediction Web App")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Build input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "gender": [gender],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "Residence_type": [Residence_type],
    "smoking_status": [smoking_status]
})

# Apply same preprocessing as training (e.g., label encoding or one-hot)
# Example if you used pd.get_dummies:
input_data = pd.get_dummies(input_data)

# Align input_data with training columns
missing_cols = set(feature_columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[feature_columns]
# Predict
prediction = model.predict(input_data)[0]

if prediction == 1:
    st.error("⚠️ High risk of stroke")
else:
    st.success("✅ Low risk of stroke")
