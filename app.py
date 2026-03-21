import streamlit as st
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np

# ================================
# ✅ LOAD PIPELINE + MODEL
# ================================
with open("models/preprocessing_pipeline.pkl", "rb") as f:
    preprocess = pickle.load(f)

model = tf.keras.models.load_model("models/ann_churn_model.h5")

# ================================
# ✅ UI
# ================================
st.title("🔥 Churn Prediction App with SHAP Explainability")

st.subheader("Enter Customer Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 100, 5)
PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "PaymentMethod",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)
MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 1000.0, 70.0)
TotalCharges = st.number_input("TotalCharges", 0.0, 10000.0, 400.0)

# ================================
# ✅ INPUT DATAFRAME
# ================================
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

# ================================
# ✅ PREDICTION + SHAP
# ================================
if st.button("Predict Churn"):

    # 🔥 PREPROCESS
    processed = preprocess.transform(input_df).astype(np.float32)

    # 🔥 PREDICT
    prob = model.predict(processed)[0][0]
    label = "WILL CHURN ❌" if prob > 0.5 else "WILL NOT CHURN ✅"

    st.write(f"Churn Probability: {prob:.4f}")
    st.write(f"Decision: {label}")

    # ================================
    # 🔥 SHAP (FINAL STABLE)
    # ================================
    st.subheader("Feature Importance (SHAP values)")

    try:
        background = np.zeros((50, processed.shape[1]))

        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(processed)

        plt.figure()
        shap.bar_plot(shap_values[0][0])

        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.warning(f"SHAP failed: {e}")