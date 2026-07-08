**🔥 Customer Churn Prediction with SHAP Explainability**

An ANN-based churn predictor that doesn't just say "this customer will leave" — it shows exactly which factors drove that prediction, using SHAP.

Built end-to-end: data cleaning → preprocessing pipeline → neural network training → explainability layer → interactive Streamlit app.


**💡 Why This Matters**

Churn prediction alone isn't enough for a business to act on — a retention team needs to know why a customer is flagged as high-risk to decide what to do about it. This project pairs a trained neural network with SHAP (SHapley Additive exPlanations) so every prediction comes with a transparent, per-feature breakdown, the same interpretability approach used in production risk and retention models.


Input: customer profile (contract type, tenure, charges, services, etc.) → Output: churn probability + a visual breakdown of which features pushed the prediction toward "will churn" or "will stay."



## 🚨 Problem
Customer churn directly impacts business revenue.  
Most models only give predictions, but businesses also need **clear reasons** behind those predictions to take action.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://github.com/Allure815/ai-churn-prediction-ann/blob/main/Demo-Churn_ANN.mp4)

---

## 🖼️ App Preview
![Churn Prediction App](https://github.com/Allure815/ai-churn-prediction-ann/blob/main/SS-Churn.png)

---


**⚙️ Key Features**


🧠 Feedforward neural network (Keras Dense 32 → 16 → 1) trained on the IBM Telco Customer Churn dataset

🔍 SHAP-based feature importance for every individual prediction, not just global model stats

📊 Churn probability score alongside the binary decision

🎛️ Full customer-profile input form (19 features: demographics, services, contract, billing) via Streamlit

🧹 Proper preprocessing pipeline (StandardScaler + OneHotEncoder via ColumnTransformer), saved and reused at inference time so training and serving stay consistent

---


## ⚙️ What This Project Does
- Takes customer details as input  
- Predicts whether the customer will churn  
- Displays probability of churn  
- Shows key factors influencing the prediction  

---

**🧠 How It Works**

-User fills in a customer's profile (tenure, contract type, services, billing info, etc.)

-Input is transformed through the same ColumnTransformer pipeline used during training

-The trained ANN outputs a churn probability

-SHAP's KernelExplainer computes per-feature contributions to that specific prediction

-Results are displayed as a probability score, a churn/no-churn decision, and a SHAP bar chart of the top influencing factors


---

**🛠️ Tech Stack**

Modeling: TensorFlow / Keras (ANN)

Preprocessing: scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder)

Explainability: SHAP

Interface: Streamlit

Dataset: IBM Telco Customer Churn (public)

---


## 📊 Use Cases
- Customer retention strategy  
- Identifying high-risk customers  
- Understanding churn drivers  
- Learning interpretable ML  

---



**▶️ Run It Locally**

bash# Clone
git clone https://github.com/Allure815/ai-churn-prediction-ann.git
cd ai-churn-prediction-ann

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py

Fill in a customer profile and click Predict Churn to see the probability and SHAP explanation.

----


**🔭 What's Next**


Wire up the SHAP explainer to use the pre-computed real-data background (shap_background.joblib) instead of a zero baseline, for more accurate feature attributions
Report test-set accuracy, precision/recall, and ROC-AUC alongside training accuracy for a complete performance picture
Experiment with additional epochs / architecture tuning to push past the current ~80% training accuracy
Add a batch-prediction mode (upload a CSV of customers, get churn scores for all of them at once)



**👤 Author**

Heeral — https://github.com/Allure815


  
