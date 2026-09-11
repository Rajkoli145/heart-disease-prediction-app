import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()

# Header
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application uses an **Ensemble Random Forest Classifier** trained on the UCI Heart Disease dataset 
to assess the risk of coronary heart disease based on clinical patient parameters.
""")
st.write("---")

# Layout columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographic & General")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
    sex_label = st.selectbox("Sex", options=["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130, step=1)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)

with col2:
    st.subheader("🩺 Symptom & Test Results")
    cp_options = {
        "Typical Angina (0)": 0,
        "Atypical Angina (1)": 1,
        "Non-anginal Pain (2)": 2,
        "Asymptomatic (3)": 3
    }
    cp_label = st.selectbox("Chest Pain Type", options=list(cp_options.keys()))
    cp = cp_options[cp_label]
    
    fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (False)", "Yes (True)"])
    fbs = 1 if "Yes" in fbs_label else 0
    
    restecg_options = {
        "Normal (0)": 0,
        "ST-T Wave Abnormality (1)": 1,
        "Left Ventricular Hypertrophy (2)": 2
    }
    restecg_label = st.selectbox("Resting ECG Results", options=list(restecg_options.keys()))
    restecg = restecg_options[restecg_label]
    
    thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=230, value=150, step=1)

with col3:
    st.subheader("🏃 Exercise & Heart Scans")
    exang_label = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
    exang = 1 if exang_label == "Yes" else 0
    
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    
    slope_options = {
        "Upsloping (0)": 0,
        "Flat (1)": 1,
        "Downsloping (2)": 2
    }
    slope_label = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_options.keys()), index=1)
    slope = slope_options[slope_label]
    
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3], index=0)
    
    thal_options = {
        "Normal (1)": 1,
        "Fixed Defect (2)": 2,
        "Reversible Defect (3)": 3
    }
    thal_label = st.selectbox("Thalassemia Status", options=list(thal_options.keys()), index=1)
    thal = thal_options[thal_label]

st.write("---")

# Prediction Button
if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
    # Prepare feature dataframe
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                               thalach, exang, oldpeak, slope, ca, thal]], 
                             columns=feature_columns)
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    prob_disease = probabilities[1] * 100
    prob_healthy = probabilities[0] * 100
    
    st.subheader("📊 Diagnostic Assessment Results")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk: Heart Disease Detected**")
            st.markdown(f"The model indicates a **{prob_disease:.1f}% probability** of heart disease presence.")
        else:
            st.success("✅ **Low Risk: No Heart Disease Detected**")
            st.markdown(f"The model indicates a **{prob_healthy:.1f}% probability** of a healthy heart profile.")
            
    with res_col2:
        st.metric(label="Risk Probability", value=f"{prob_disease:.1f}%", delta=f"{prob_disease - 50:.1f}% vs baseline")
        st.progress(int(prob_disease))

    st.info("""
    **Clinical Note:** Random Forest ensemble predictions are intended for preliminary screening and educational purposes. 
    Consult a licensed cardiologist for comprehensive diagnostic evaluation.
    """)
