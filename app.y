```python
import streamlit as st
import pickle
import numpy as np

# Page settings
st.set_page_config(page_title="Churn Prediction App", layout="centered")

# Title
st.title("📊 Customer Churn Prediction")
st.write("Fill customer details to predict whether they will churn or stay.")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Sidebar inputs
st.sidebar.header("Input Customer Data")

age = st.sidebar.slider("Age", 18, 100, 30)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=5000.0)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)

# Predict button
if st.sidebar.button("Predict"):

    # Convert input into array
    input_data = np.array([[age, balance, tenure]])

    # Prediction
    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer will stay")

# Footer
st.markdown("---")
st.caption("Built using Streamlit 🚀")
```
