import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model

model = pickle.load(open("model.pkl", "rb"))

# Page config

st.set_page_config(page_title="Churn Predictor", layout="wide")

# Title

st.title("📊 Customer Churn Prediction Dashboard")

# Sidebar Inputs

st.sidebar.header("🧾 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72)
PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet"])
Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox(
"Payment Method",
["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0)

# -------------------------

# Encoding Function

# -------------------------
def encode():
    data = [
        1 if gender == "Male" else 0,
        1 if SeniorCitizen == "Yes" else 0,
        1 if Partner == "Yes" else 0,
        1 if Dependents == "Yes" else 0,
        tenure,
        1 if PhoneService == "Yes" else 0,
        {"No": 0, "Yes": 1, "No phone service": 2}[MultipleLines],
        {"DSL": 0, "Fiber optic": 1, "No": 2}[InternetService],
        {"No": 0, "Yes": 1, "No internet": 2}[OnlineSecurity],
        {"No": 0, "Yes": 1, "No internet": 2}[OnlineBackup],
        {"No": 0, "Yes": 1, "No internet": 2}[DeviceProtection],
        {"No": 0, "Yes": 1, "No internet": 2}[TechSupport],
        {"No": 0, "Yes": 1, "No internet": 2}[StreamingTV],
        {"No": 0, "Yes": 1, "No internet": 2}[StreamingMovies],
        {"Month-to-month": 0, "One year": 1, "Two year": 2}[Contract],
        1 if PaperlessBilling == "Yes" else 0,
        {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}[PaymentMethod],
        MonthlyCharges,
        TotalCharges
    ]
    return np.array([data])

# -------------------------

# Prediction

# -------------------------

if st.sidebar.button("🚀 Predict"):

input_data = encode()

prediction = model.predict(input_data)
prob = model.predict_proba(input_data)

churn_prob = prob[0][1] * 100
stay_prob = prob[0][0] * 100

st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer will STAY")

with col2:
    st.metric("Churn Probability", f"{round(churn_prob, 2)} %")

# Chart
st.subheader("📊 Probability Comparison")

chart_data = pd.DataFrame({
    "Status": ["Stay", "Churn"],
    "Probability": [stay_prob, churn_prob]
})

st.bar_chart(chart_data.set_index("Status"))

# Table
st.subheader("🥧 Probability Distribution")
st.write(chart_data)

# Insights
st.subheader("🧠 Customer Insights")

insights = []

if tenure < 12:
    insights.append("⚠️ New customers are more likely to churn")
if MonthlyCharges > 70:
    insights.append("💸 High monthly charges increase churn risk")
if Contract == "Month-to-month":
    insights.append("📉 Month-to-month contracts have higher churn")
if TechSupport == "No":
    insights.append("🛠️ Lack of tech support increases churn")

if len(insights) == 0:
    insights.append("✅ Customer profile looks stable")

for i in insights:
    st.write(i)

# Download
st.subheader("📥 Download Report")

report = pd.DataFrame({
    "Prediction": ["Churn" if prediction[0] == 1 else "Stay"],
    "Churn Probability (%)": [round(churn_prob, 2)],
    "Tenure": [tenure],
    "Monthly Charges": [MonthlyCharges],
    "Contract": [Contract]
})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Report as CSV",
    data=csv,
    file_name="churn_report.csv",
    mime="text/csv"
)
