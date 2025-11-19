# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
from datetime import datetime

# plotting
import matplotlib.pyplot as plt

# PDF generation (optional; add "fpdf" to requirements.txt for PDF export)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Kraljic Matrix Classifier (Upgraded)", layout="wide")
st.title("🧠 Kraljic Matrix Classification — Upgraded")
st.markdown(
    "A more realistic procurement tool — India-focused regions, batch CSV upload, visualizations, "
    "recommendations, and downloadable reports."
)

# ---------------------------
# Model loading & expected features
# ---------------------------
MODEL_PATH = "naive_bayes_model.pkl"
MODEL_COLUMNS_PATH = "model_columns.pkl"  # optional (saved during training)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Put it in the same folder as app.py and restart.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Determine the expected feature names (prefer saved list)
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
        model_columns = joblib.load(MODEL_COLUMNS_PATH)
        # ensure it's a list-like
        model_columns = list(model_columns)
    except Exception:
        model_columns = default_model_columns
else:
    model_columns = default_model_columns

# ---------------------------
# Sidebar: Input / Batch Upload
# ---------------------------
st.sidebar.header("Input / Batch Options")

mode = st.sidebar.radio("Mode", ["Single item", "Batch upload (CSV)"])

# Common: Indian-focused region list (but NOTE: region is NOT used for prediction unless model trained on it)
REGIONS = [
    "Maharashtra", "Gujarat", "Karnataka", "Delhi NCR", "Tamil Nadu",
    "West Bengal", "Rajasthan", "Uttar Pradesh", "Kerala", "Punjab",
    "China", "Bangladesh", "GCC", "USA", "Europe", "Other"
]

if mode == "Single item":
    # Sidebar: single item inputs
    st.sidebar.subheader("Procurement Item Details (Single)")

    lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=0, max_value=3650, value=30, step=1)
    order_volume = st.sidebar.number_input("Order Volume (Units)", min_value=1, max_value=10_000_000, value=500, step=1)
    cost_per_unit = st.sidebar.number_input("Cost per Unit (₹)", min_value=0.0, max_value=10_000_000.0, value=250.0, format="%.2f")
    supply_risk = st.sidebar.slider("Supply Risk Score (1=Low, 5=High)", 1, 5, 3)
    profit_impact = st.sidebar.slider("Profit Impact Score (1=Low, 5=High)", 1, 5, 3)
    env_impact = st.sidebar.slider("Environmental Impact Score (1=Low, 5=High)", 1, 5, 2)
    region = st.sidebar.selectbox("Supplier Region (for reporting)", REGIONS)
    single_source = st.sidebar.selectbox("Single Source Risk?", ["Yes", "No"])

    # quick input validation settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Validation rules**")
    st.sidebar.write("- Lead time should be realistic (0–3650 days)")
    st.sidebar.write("- Cost & order volume should be non-negative")

else:
    st.sidebar.subheader("Batch CSV Upload")
    st.sidebar.markdown(
        "Upload a CSV with columns matching (or containing) at least these: "
        "`Lead_Time_Days`, `Order_Volume_Units`, `Cost_per_Unit`, "
        "`Supply_Risk_Score`, `Profit_Impact_Score`, `Environmental_Impact`, `Single_Source_Risk`."
    )
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    sample_button = st.sidebar.button("Download sample CSV")

    if sample_button:
        # create sample csv
        sample_df = pd.DataFrame({
            "Lead_Time_Days": [30, 90],
            "Order_Volume_Units": [500, 10000],
            "Cost_per_Unit": [250.0, 12.5],
            "Supply_Risk_Score": [3, 4],
            "Profit_Impact_Score": [3, 5],
            "Environmental_Impact": [2, 4],
            "Single_Source_Risk": [1, 0],
            "Supplier_Region": ["Maharashtra", "China"],
            "Product_Name": ["Item A", "Item B"]
        })
        csv_bytes = sample_df.to_csv(index=False).encode()
        st.sidebar.download_button("Download sample CSV", data=csv_bytes, file_name="sample_kraljic_upload.csv", mime="text/csv")

# ---------------------------
# Utility functions
# ---------------------------
def validate_single_input(ld, vol, cost):
    errors = []
    if ld < 0:
        errors.append("Lead time cannot be negative.")
    if vol <= 0:
        errors.append("Order volume must be > 0.")
    if cost <= 0:
        errors.append("Cost per unit must be > 0.")
    return errors

def prepare_input_df(df_row):
    """
    Ensure df_row (DataFrame with single row or many rows) aligns with model_columns.
    - Drop Supplier_Region if present
    - Convert Single_Source_Risk Yes/No to 1/0 if necessary
    - Reindex columns to model_columns (fill missing with 0)
    """
    df = df_row.copy()

    if "Supplier_Region" in df.columns:
        df = df.drop(columns=["Supplier_Region"], errors="ignore")

    # Convert Single_Source_Risk values
    if "Single_Source_Risk" in df.columns:
        df["Single_Source_Risk"] = df["Single_Source_Risk"].map({ "Yes": 1, "No": 0, "yes":1, "no":0 }).fillna(df["Single_Source_Risk"])
        # If it's boolean True/False, convert to int
        if df["Single_Source_Risk"].dtype == bool:
            df["Single_Source_Risk"] = df["Single_Source_Risk"].astype(int)

    # Ensure numeric columns exist, fill missing with 0 (or better: median) — Here we fill with 0 to avoid errors
    final_df = df.reindex(columns=model_columns, fill_value=0)
    return final_df

def predict_and_attach(df_inputs):
    """
    df_inputs: prepared DataFrame aligned to model_columns
    returns: predictions, probabilities (if available)
    """
    preds = model.predict(df_inputs)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df_inputs)
        except Exception:
            proba = None
    return preds, proba

def category_color(cat):
    """Return a color badge and emoji for category"""
    mapping = {
        "Strategic": ("🔴", "#ff4b4b"),
        "Leverage": ("🔵", "#4b7bff"),
        "Bottleneck": ("🟡", "#ffcc00"),
        "Non-Critical": ("🟢", "#2ecc71")
    }
    return mapping.get(cat, ("⚪", "#999999"))

def recommendations_for_category(cat):
    """Actionable recommendations for procurement teams"""
    recs = {
        "Strategic": [
            "Establish long-term partnerships and supplier development plans.",
            "Consider joint forecasting, collaborative risk mitigation and safety stocks.",
            "Prioritize supplier performance monitoring and contingency planning."
        ],
        "Leverage": [
            "Run competitive bidding and volume consolidation to reduce unit costs.",
            "Leverage multiple suppliers to negotiate better terms.",
            "Use dynamic sourcing and reverse auctions where possible."
        ],
        "Bottleneck": [
            "Diversify suppliers and build redundancy for critical inputs.",
            "Consider safety stocks and increase monitoring of supplier health.",
            "Evaluate alternative materials or redesign to reduce dependency."
        ],
        "Non-Critical": [
            "Automate procurement and use catalog management to reduce PO overhead.",
            "Standardize items and use long-term contracts for efficiency.",
            "Consider vendor-managed inventory for low-impact items."
        ]
    }
    return recs.get(cat, ["No specific recommendation available."])

def create_pdf_report(single_input_row, predicted_category, proba=None):
    """
    Create a simple single-page PDF summarizing the input & prediction.
    Returns bytes of PDF.
    """
    if not FPDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Kraljic Matrix — Procurement Risk Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)

    # Input Table
    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0, 8, "Input Summary:", ln=True)
    pdf.set_font("Arial", size=10)
    for col, val in single_input_row.items():
        pdf.cell(0, 7, f"- {col}: {val}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0, 8, f"Predicted Kraljic Category: {predicted_category}", ln=True)
    pdf.set_font("Arial", size=10)
    if proba is not None:
        pdf.ln(4)
        pdf.cell(0, 8, "Predicted probabilities:", ln=True)
        labels = list(model.classes_)
        for lbl, p in zip(labels, proba):
            pdf.cell(0, 7, f"- {lbl}: {p:.3f}", ln=True)

    # Recommendations
    pdf.ln(6)
    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0, 8, "Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    recs = recommendations_for_category(predicted_category)
    for r in recs:
        pdf.multi_cell(0, 7, f"- {r}")

    return pdf.output(dest="S").encode("latin-1")

# ---------------------------
# MAIN: Single Mode
# ---------------------------
if mode == "Single item":
    # Input validation
    validation_errors = validate_single_input(lead_time, order_volume, cost_per_unit)
    if validation_errors:
        for err in validation_errors:
            st.error(err)
        st.stop()

    # Prepare dataframe for one row
    input_df = pd.DataFrame({
        "Lead_Time_Days": [int(lead_time)],
        "Order_Volume_Units": [int(order_volume)],
        "Cost_per_Unit": [float(cost_per_unit)],
        "Supply_Risk_Score": [int(supply_risk)],
        "Profit_Impact_Score": [int(profit_impact)],
        "Environmental_Impact": [int(env_impact)],
        "Single_Source_Risk": [1 if single_source == "Yes" else 0],
        # keep region & product name for report only (not used in prediction)
        "Supplier_Region": [region]
    })

    st.subheader("🔍 Input Summary")
    st.table(input_df.T)

    # Predict
    if st.button("Predict Category"):
        try:
            prepared = prepare_input_df(input_df)
            preds, proba = predict_and_attach(prepared)
            category = preds[0]

            emoji, color = category_color(category)

            # show colored result
            st.markdown(f"### {emoji} Predicted Kraljic Category: **{category}**")
            st.markdown(f"<div style='background:{color};height:8px;border-radius:4px'></div>", unsafe_allow_html=True)

            # show probabilities if available
            if proba is not None:
                proba_series = pd.Series(proba[0], index=model.classes_).sort_values(ascending=False)
                st.subheader("Model confidence (probabilities)")
                st.bar_chart(proba_series)

            # show quadrant / heatmap (Profit Impact vs Supply Risk)
            st.subheader("Kraljic Quadrant (Profit Impact vs Supply Risk)")
            fig, ax = plt.subplots(figsize=(6,6))
            # create 5x5 grid for 1-5 scales
            ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
            ax.set_xlabel("Profit Impact Score (1=Low → 5=High)")
            ax.set_ylabel("Supply Risk Score (1=Low → 5=High)")
            ax.set_xticks([1,2,3,4,5]); ax.set_yticks([1,2,3,4,5])
            # draw quadrant lines at 3 (midpoint)
            ax.axvline(3, color="grey", linestyle="--")
            ax.axhline(3, color="grey", linestyle="--")
            # annotate quadrants
            ax.text(1.2,4.2,"Non-Critical", fontsize=10, color="green")
            ax.text(3.2,4.2,"Leverage", fontsize=10, color="blue")
            ax.text(1.2,1.2,"Bottleneck", fontsize=10, color="orange")
            ax.text(3.2,1.2,"Strategic", fontsize=10, color="red")
            # plot the point
            px = input_df["Profit_Impact_Score"].iloc[0]
            py = input_df["Supply_Risk_Score"].iloc[0]
            ax.scatter(px, py, s=150, c="black", marker="X")
            ax.set_title("Kraljic Matrix (point shows the item position)")
            st.pyplot(fig)

            # Recommendations
            st.subheader("Recommended Actions")
            recs = recommendations_for_category(category)
            for r in recs:
                st.write("•", r)

            # Downloadable report (PDF or CSV fallback)
            if FPDF_AVAILABLE:
                pdf_bytes = create_pdf_report(input_df.iloc[0].to_dict(), category, proba[0] if proba is not None else None)
                if pdf_bytes:
                    st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"kraljic_report_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
            else:
                st.info("To enable PDF report download, add 'fpdf' to requirements.txt. For now, you can download CSV of the input.")
                csv_bytes = input_df.to_csv(index=False).encode()
                st.download_button("Download input as CSV", data=csv_bytes, file_name="kraljic_input.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------
# MAIN: Batch Mode
# ---------------------------
else:
    st.subheader("Batch Predictions (CSV)")
    if uploaded_file is None:
        st.info("Upload a CSV file to run batch predictions. Use the sample CSV if you need a template.")
    else:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("First 5 rows of uploaded file:")
            st.dataframe(batch_df.head())

            # Prepare data
            prepared = prepare_input_df(batch_df)
            preds, proba = predict_and_attach(prepared)
            batch_df["Predicted_Kraljic_Category"] = preds

            # If probabilities available, attach top probability
            if proba is not None:
                top_proba = np.max(proba, axis=1)
                batch_df["Prediction_Confidence"] = top_proba

            st.success(f"Predicted {len(batch_df)} rows.")
            st.dataframe(batch_df.head())

            # Download results
            out_csv = batch_df.to_csv(index=False).encode()
            st.download_button("Download predictions (CSV)", data=out_csv, file_name="kraljic_batch_predictions.csv", mime="text/csv")

            # Simple summary chart
            st.subheader("Prediction Distribution")
            dist = batch_df["Predicted_Kraljic_Category"].value_counts()
            st.bar_chart(dist)

            # Option: Save to file in app folder (server)
            if st.button("Save results to server (predictions.csv)"):
                batch_df.to_csv("predictions.csv", index=False)
                st.success("Saved as predictions.csv in app directory.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------------------
# Footer: Notes & Requirements
# ---------------------------
st.markdown("---")
st.caption(
    "Notes: This app expects the model 'naive_bayes_model.pkl' in the same folder. "
    "If you want PDF reports, add 'fpdf' to requirements.txt (pip install fpdf). "
    "Make sure model was trained on features: " + ", ".join(model_columns)
)
