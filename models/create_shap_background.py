import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 🔥 FIX: Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows with missing values (same logic used in training)
df.dropna(inplace=True)

# Separate features
X = df.drop("Churn", axis=1)

# Load preprocessing pipeline
preprocess = joblib.load("preprocessing_pipeline.joblib")

# Transform
X_processed = preprocess.transform(X).astype("float32")

# Save SHAP background
joblib.dump(X_processed[:50], "shap_background.joblib")

print("✅ shap_background.joblib created successfully")
