import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------
# Load model and preprocessors
# ---------------------------
model_data = joblib.load("kraljic_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
label_encoder = model_data["label_encoder"]

st.set_page_config(page_title="Kraljic Matrix Classifier", layout="centered")

# ---------------------------
# UI Title
# ---------------------------
st.title("🧠 Kraljic Matrix Classification App")
st.write("""
This app predicts the **Kraljic Category** of a procurement item  
based on supply chain parameters such as cost, risk, and lead time.
""")

# ---------------------------
# Sidebar Input Section
# ---------------------------
st.sidebar.header("Enter Procurement Item Details")

lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=1, max_value=365, value=30)
order_volume = st.sidebar.number_input("Order Volume (Units)", min_value=1, max_value=10000, value=500)
cost_per_unit = st.sidebar.number_input("Cost per Unit ($)", min_value=0.1, max_value=10000.0, value=250.0)
supply_risk = st.sidebar.slider("Supply Risk Score (1 = Low, 5 = High)", 1, 5, 3)
profit_impact = st.sidebar.slider("Profit Impact Score (1 = Low, 5 = High)", 1, 5, 3)
env_impact = st.sidebar.slider("Environmental Impact Score (1 = Low, 5 = High)", 1, 5, 2)

region = st.sidebar.selectbox(
    "Supplier Region", ["Asia", "Europe", "Africa", "North America", "South America"]
)
single_source = st.sidebar.selectbox("Single Source Risk?", ["Yes", "No"])

# ---------------------------
# Prepare input data
# ---------------------------
input_df = pd.DataFrame(
    {
        "Lead_Time_Days": [lead_time],
        "Order_Volume_Units": [order_volume],
        "Cost_per_Unit": [cost_per_unit],
        "Supply_Risk_Score": [supply_risk],
        "Profit_Impact_Score": [profit_impact],
        "Environmental_Impact": [env_impact],
        "Supplier_Region": [region],
        "Single_Source_Risk": [single_source],
    }
)

# Convert categorical to numeric if necessary
# (If model expects numeric/encoded features)
try:
    X_scaled = scaler.transform(input_df.select_dtypes(include=np.number))
except Exception:
    X_scaled = scaler.transform(np.array(input_df).reshape(1, -1))

# ---------------------------
# Prediction
# ---------------------------
if st.sidebar.button("Predict Category"):
    prediction = model.predict(X_scaled)
    category = label_encoder.inverse_transform(prediction)[0]

    st.success(f"### 🔍 Predicted Kraljic Category: **{category}**")

    st.info("""
    **Category Meaning:**
    - Strategic: High risk, high profit impact — build partnerships.
    - Leverage: High profit, low risk — use negotiation power.
    - Bottleneck: High risk, low profit — reduce dependency.
    - Non-Critical: Low risk, low profit — streamline efficiency.
    """)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed by **Dikesh Chavhan** | Final Year Engineering Project (2025)")
