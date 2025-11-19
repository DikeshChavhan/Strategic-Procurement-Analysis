# app.py — Cleaned Version (No PDF)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Kraljic Matrix Classifier", layout="wide")
st.title("🧠 Kraljic Matrix Classification — Simplified & Cleaned")
st.markdown(
    "A practical procurement classification app using the **Kraljic Matrix**. "
    "Includes Indian supplier regions, batch CSV upload, charts, recommendations, "
    "and a clean UI—no PDF generation."
)

# ---------------------------
# Model loading
# ---------------------------
MODEL_PATH = "naive_bayes_model.pkl"
MODEL_COLUMNS_PATH = "model_columns.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found in app folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

default_model_columns = [
    "Lead_Time_Days",
    "Order_Volume_Units",
    "Cost_per_Unit",
    "Supply_Risk_Score",
    "Profit_Impact_Score",
    "Environmental_Impact",
    "Single_Source_Risk"
]

if os.path.exists(MODEL_COLUMNS_PATH):
    try:
        model_columns = list(joblib.load(MODEL_COLUMNS_PATH))
    except:
        model_columns = default_model_columns
else:
    model_columns = default_model_columns

# ---------------------------
# Sidebar Input Settings
# ---------------------------
st.sidebar.header("Input / Batch Options")
mode = st.sidebar.radio("Mode", ["Single item", "Batch upload (CSV)"])

REGIONS = [
    "Maharashtra", "Gujarat", "Karnataka", "Delhi NCR", "Tamil Nadu",
    "West Bengal", "Rajasthan", "Uttar Pradesh", "Kerala", "Punjab",
    "China", "Bangladesh", "GCC", "USA", "Europe", "Other"
]

# ---------------------------
# Helper Functions
# ---------------------------
def validate_single_input(ld, vol, cost):
    errors = []
    if ld < 0:
        errors.append("Lead time must be ≥ 0.")
    if vol <= 0:
        errors.append("Order volume must be > 0.")
    if cost <= 0:
        errors.append("Cost per unit must be > 0.")
    return errors


def prepare_input_df(df_row):
    df = df_row.copy()

    if "Supplier_Region" in df.columns:
        df = df.drop(columns=["Supplier_Region"], errors="ignore")

    # Convert Yes/No to 1/0
    if "Single_Source_Risk" in df.columns:
        df["Single_Source_Risk"] = df["Single_Source_Risk"].map(
            {"Yes": 1, "No": 0}
        ).fillna(df["Single_Source_Risk"])

    # Align columns
    final_df = df.reindex(columns=model_columns, fill_value=0)
    return final_df


def predict_and_attach(df_inputs):
    preds = model.predict(df_inputs)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df_inputs)
        except:
            pass
    return preds, proba


def category_color(cat):
    mapping = {
        "Strategic": ("🔴", "#ff4b4b"),
        "Leverage": ("🔵", "#4b7bff"),
        "Bottleneck": ("🟡", "#ffcc00"),
        "Non-Critical": ("🟢", "#2ecc71")
    }
    return mapping.get(cat, ("⚪", "#999999"))


def recommendations_for_category(cat):
    recs = {
        "Strategic": [
            "Build long-term partnerships.",
            "Create joint forecasting and risk mitigation plans.",
            "Invest in supplier development."
        ],
        "Leverage": [
            "Use competitive bidding.",
            "Optimize negotiation strategies.",
            "Consolidate volumes for better pricing."
        ],
        "Bottleneck": [
            "Develop backup suppliers.",
            "Increase safety stock.",
            "Explore substitute materials."
        ],
        "Non-Critical": [
            "Automate purchasing (catalog buying).",
            "Focus on process efficiency.",
            "Use long-term contracts for low-value items."
        ]
    }
    return recs.get(cat, ["No recommendations available."])

# ---------------------------
# SINGLE ITEM MODE
# ---------------------------
if mode == "Single item":

    st.sidebar.subheader("Procurement Item Details")

    lead_time = st.sidebar.number_input("Lead Time (Days)", 0, 3650, 30)
    order_volume = st.sidebar.number_input("Order Volume (Units)", 1, 10_000_000, 500)
    cost_per_unit = st.sidebar.number_input("Cost per Unit (₹)", 0.1, 10_000_000.0, 250.0)
    supply_risk = st.sidebar.slider("Supply Risk Score", 1, 5, 3)
    profit_impact = st.sidebar.slider("Profit Impact Score", 1, 5, 3)
    env_impact = st.sidebar.slider("Environmental Impact Score", 1, 5, 2)
    single_source = st.sidebar.selectbox("Single Source Risk?", ["Yes", "No"])
    region = st.sidebar.selectbox("Supplier Region (India-focused)", REGIONS)

    # validation
    errors = validate_single_input(lead_time, order_volume, cost_per_unit)
    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    # prepare DF
    input_df = pd.DataFrame({
        "Lead_Time_Days": [lead_time],
        "Order_Volume_Units": [order_volume],
        "Cost_per_Unit": [cost_per_unit],
        "Supply_Risk_Score": [supply_risk],
        "Profit_Impact_Score": [profit_impact],
        "Environmental_Impact": [env_impact],
        "Single_Source_Risk": [1 if single_source == "Yes" else 0],
        "Supplier_Region": [region]  # used for display only
    })

    st.subheader("🔍 Input Summary")
    st.table(input_df.T)

    if st.button("Predict Category"):
        try:
            prepared = prepare_input_df(input_df)
            preds, proba = predict_and_attach(prepared)
            category = preds[0]

            emoji, color = category_color(category)
            st.markdown(f"### {emoji} Predicted Category: **{category}**")
            st.markdown(f"<div style='background:{color};height:8px;border-radius:4px'></div>", unsafe_allow_html=True)

            # probabilities
            if proba is not None:
                st.subheader("Model Confidence")
                proba_series = pd.Series(proba[0], index=model.classes_)
                st.bar_chart(proba_series)

            # quadrant chart
            st.subheader("Kraljic Quadrant Position")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Profit Impact")
            ax.set_ylabel("Supply Risk")
            ax.axvline(3, color="grey", linestyle="--")
            ax.axhline(3, color="grey", linestyle="--")

            ax.text(1, 4.5, "Non-Critical", color="green")
            ax.text(3.2, 4.5, "Leverage", color="blue")
            ax.text(1, 1, "Bottleneck", color="orange")
            ax.text(3.2, 1, "Strategic", color="red")

            ax.scatter(profit_impact, supply_risk, s=150, c="black", marker="X")
            st.pyplot(fig)

            # recommendations
            st.subheader("Recommended Actions")
            for r in recommendations_for_category(category):
                st.write("•", r)

            # simple CSV download
            csv_bytes = input_df.to_csv(index=False).encode()
            st.download_button("Download input data (CSV)", data=csv_bytes,
                               file_name="kraljic_input.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------
# BATCH CSV MODE
# ---------------------------
else:
    st.subheader("Batch Predictions")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to start batch processing.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded data preview:")
            st.dataframe(df.head())

            prepared = prepare_input_df(df)
            preds, proba = predict_and_attach(prepared)

            df["Predicted_Kraljic_Category"] = preds

            if proba is not None:
                df["Prediction_Confidence"] = np.max(proba, axis=1)

            st.success(f"Predicted {len(df)} rows successfully.")
            st.dataframe(df.head())

            out_csv = df.to_csv(index=False).encode()
            st.download_button("Download predictions (CSV)", data=out_csv,
                               file_name="kraljic_predictions.csv",
                               mime="text/csv")

            st.subheader("Category Distribution")
            st.bar_chart(df["Predicted_Kraljic_Category"].value_counts())

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(
    "Kraljic Matrix Classifier — Clean Version • "
    f"Model expects features: {', '.join(model_columns)}"
)
