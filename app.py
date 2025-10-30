import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------
# App Configuration
# ---------------------------
st.set_page_config(page_title="Kraljic Matrix Classifier", layout="centered")

st.title("🧠 Kraljic Matrix Classification App")
st.write("""
This web app predicts the **Kraljic Category** (Strategic, Leverage, Bottleneck, or Non-Critical)  
based on procurement attributes like cost, risk, and lead time.
""")

# ---------------------------
# Load Model Safely
# ---------------------------
MODEL_PATH = "naive_bayes_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'naive_bayes_model.pkl' not found. Please ensure it is in the same directory as `app.py`.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Enter Procurement Details")

# ---------------------------
# Sidebar Inputs
# ---------------------------
lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=1, max_value=365, value=30)
order_volume = st.sidebar.number_input("Order Volume (Units)", min_value=1, max_value=10000, value=500)
cost_per_unit = st.sidebar.number_input("Cost per Unit ($)", min_value=0.1, max_value=10000.0, value=250.0)
supply_risk = st.sidebar.slider("Supply Risk Score (1 = Low, 5 = High)", 1, 5, 3)
profit_impact = st.sidebar.slider("Profit Impact Score (1 = Low, 5 = High)", 1, 5, 3)
env_impact = st.sidebar.slider("Environmental Impact Score (1 = Low, 5 = High)", 1, 5, 2)

region = st.sidebar.selectbox("Supplier Region", ["Asia", "Europe", "Africa", "North America", "South America"])
single_source = st.sidebar.selectbox("Single Source Risk?", ["Yes", "No"])

# ---------------------------
# Prepare input data
# ---------------------------
input_data = pd.DataFrame({
    "Lead_Time_Days": [lead_time],
    "Order_Volume_Units": [order_volume],
    "Cost_per_Unit": [cost_per_unit],
    "Supply_Risk_Score": [supply_risk],
    "Profit_Impact_Score": [profit_impact],
    "Environmental_Impact": [env_impact],
    "Supplier_Region": [region],
    "Single_Source_Risk": [single_source]
})

st.subheader("🔍 Input Summary")
st.write(input_data)

# ---------------------------
# Predict Button
# ---------------------------
if st.sidebar.button("Predict Category"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"### 🧩 Predicted Kraljic Category: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")

    st.info("""
    **Kraljic Matrix Reference:**
    - **Strategic**: High risk, high profit → Long-term partnerships  
    - **Leverage**: Low risk, high profit → Maximize negotiation power  
    - **Bottleneck**: High risk, low profit → Diversify sources  
    - **Non-Critical**: Low risk, low profit → Optimize efficiency  
    """)

# ---------------------------
# Footer
# ---------------------------
st.caption("Developed by **Dikesh Chavhan** | Strategic Procurement ML Project (2025)")
