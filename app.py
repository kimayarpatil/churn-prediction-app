import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

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

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 1)

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
    min_value=0.0,
    value=50.0,
    step=0.01
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0,
    step=0.01
)

# -------------------------
# Encoding Function
# -------------------------
def encode_features():
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
# Main Prediction Section
# -------------------------
if st.sidebar.button("🚀 Predict Churn", use_container_width=True):
    with st.spinner("🔮 Predicting..."):
        input_data = encode_features()
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        churn_prob = probabilities[1] * 100  # Class 1 = Churn
        stay_prob = probabilities[0] * 100   # Class 0 = Stay

        # -------------------------
        # Prediction Result
        # -------------------------
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **Customer is likely to CHURN**")
            else:
                st.success("✅ **Customer will STAY**")
        
        with col2:
            st.metric("Churn Probability", f"{churn_prob:.1f}%")
        
        with col3:
            st.metric("Stay Probability", f"{stay_prob:.1f}%")

        # -------------------------
        # Charts
        # -------------------------
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Probability Comparison")
            chart_data = pd.DataFrame({
                "Status": ["Stay", "Churn"],
                "Probability": [stay_prob, churn_prob]
            })
            st.bar_chart(chart_data.set_index("Status"))

        with col2:
            st.subheader("🥧 Probability Distribution")
            st.dataframe(chart_data.style.format({"Probability": "{:.1f}%"}))

        # -------------------------
        # Customer Insights
        # -------------------------
        st.subheader("🧠 Key Insights")
        
        insights = []
        col1, col2 = st.columns(2)
        
        with col1:
            if tenure < 12:
                insights.append("🆕 **Low tenure** (<12 months)")
            if monthly_charges > 70:
                insights.append("💰 **High monthly charges** (> $70)")
            if contract == "Month-to-month":
                insights.append("📅 **Month-to-month contract**")
            if paperless_billing == "Yes":
                insights.append("📄 **Paperless billing**")
        
        with col2:
            if tech_support == "No" and internet_service != "No":
                insights.append("❌ **No tech support**")
            if multiple_lines == "Yes":
                insights.append("📞 **Multiple lines**")
            if senior_citizen == "Yes":
                insights.append("👴 **Senior citizen**")

        if not insights:
            st.success("✅ **Low risk profile** - All factors favorable")
        else:
            for insight in insights:
                st.warning(insight)

        # -------------------------
        # Summary Table & Download
        # -------------------------
        st.subheader("📋 Customer Summary")
        
        summary_data = {
            "Feature": ["Prediction", "Churn Probability", "Tenure (months)", 
                       "Monthly Charges", "Contract", "Payment Method"],
            "Value": [
                "Churn" if prediction == 1 else "Stay",
                f"{churn_prob:.1f}%",
                f"{tenure}",
                f"${monthly_charges:.2f}",
                contract,
                payment_method
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download button
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Report",
            data=csv,
            file_name=f"churn_report_{tenure}_{monthly_charges:.0f}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👈 Enter customer details in the sidebar and click **Predict Churn**")
    
    # Show default info
    st.subheader("ℹ️ How to use:")
    st.markdown("""
    1. **Fill all customer details** in the sidebar
    2. **Click "Predict Churn"** button
    3. **Review prediction** and insights
    4. **Download report** for records
    """)
