import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------

# Load Model

# -------------------------

model = pickle.load(open("model.pkl", "rb"))

# -------------------------

# Page Config

# -------------------------

st.set_page_config(
page_title="Churn Predictor",
layout="wide"
)

# -------------------------

# Title

# -------------------------

st.title("📊 Customer Churn Prediction Dashboard")

# -------------------------

# Sidebar Inputs

# -------------------------

st.sidebar.header("🧾 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72)

phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox(
"Multiple Lines",
["No", "Yes", "No phone service"]
)

internet_service = st.sidebar.selectbox(
"Internet Service",
["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
"Online Security",
["No", "Yes", "No internet"]
)

online_backup = st.sidebar.selectbox(
"Online Backup",
["No", "Yes", "No internet"]
)

device_protection = st.sidebar.selectbox(
"Device Protection",
["No", "Yes", "No internet"]
)

tech_support = st.sidebar.selectbox(
"Tech Support",
["No", "Yes", "No internet"]
)

streaming_tv = st.sidebar.selectbox(
"Streaming TV",
["No", "Yes", "No internet"]
)

streaming_movies = st.sidebar.selectbox(
"Streaming Movies",
["No", "Yes", "No internet"]
)

contract = st.sidebar.selectbox(
"Contract Type",
["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
"Paperless Billing",
["No", "Yes"]
)

payment_method = st.sidebar.selectbox(
"Payment Method",
["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

monthly_charges = st.sidebar.number_input(
"Monthly Charges",
min_value=0.0
)

total_charges = st.sidebar.number_input(
"Total Charges",
min_value=0.0
)

# -------------------------

# Encoding Function

# -------------------------

def encode():
data = [
1 if gender == "Male" else 0,
1 if senior_citizen == "Yes" else 0,
1 if partner == "Yes" else 0,
1 if dependents == "Yes" else 0,
tenure,
1 if phone_service == "Yes" else 0,
{"No": 0, "Yes": 1, "No phone service": 2}[multiple_lines],
{"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
{"No": 0, "Yes": 1, "No internet": 2}[online_security],
{"No": 0, "Yes": 1, "No internet": 2}[online_backup],
{"No": 0, "Yes": 1, "No internet": 2}[device_protection],
{"No": 0, "Yes": 1, "No internet": 2}[tech_support],
{"No": 0, "Yes": 1, "No internet": 2}[streaming_tv],
{"No": 0, "Yes": 1, "No internet": 2}[streaming_movies],
{"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
1 if paperless_billing == "Yes" else 0,
{
"Electronic check": 0,
"Mailed check": 1,
"Bank transfer": 2,
"Credit card": 3
}[payment_method],
monthly_charges,
total_charges
]


return np.array([data])


# -------------------------

# Prediction

# -------------------------

if st.sidebar.button("🚀 Predict"):

input_data = encode()

prediction = model.predict(input_data)
probabilities = model.predict_proba(input_data)

churn_prob = probabilities[0][1] * 100
stay_prob = probabilities[0][0] * 100

st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer will STAY")

with col2:
    st.metric(
        "Churn Probability",
        f"{round(churn_prob, 2)} %"
    )


# -------------------------
# Chart
# -------------------------
st.subheader("📊 Probability Comparison")

chart_data = pd.DataFrame({
    "Status": ["Stay", "Churn"],
    "Probability": [stay_prob, churn_prob]
})

st.bar_chart(chart_data.set_index("Status"))


# -------------------------
# Table
# -------------------------
st.subheader("🥧 Probability Distribution")
st.write(chart_data)


# -------------------------
# Insights
# -------------------------
st.subheader("🧠 Customer Insights")

insights = []

if tenure < 12:
    insights.append("⚠️ New customers are more likely to churn")

if monthly_charges > 70:
    insights.append("💸 High monthly charges increase churn risk")

if contract == "Month-to-month":
    insights.append("📉 Month-to-month contracts have higher churn")

if tech_support == "No":
    insights.append("🛠️ Lack of tech support increases churn")

if not insights:
    insights.append("✅ Customer profile looks stable")

for insight in insights:
    st.write(insight)


# -------------------------
# Download Report
# -------------------------
st.subheader("📥 Download Report")

report = pd.DataFrame({
    "Prediction": [
        "Churn" if prediction[0] == 1 else "Stay"
    ],
    "Churn Probability (%)": [
        round(churn_prob, 2)
    ],
    "Tenure": [tenure],
    "Monthly Charges": [monthly_charges],
    "Contract": [contract]
})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Report as CSV",
    data=csv,
    file_name="churn_report.csv",
    mime="text/csv"
)
