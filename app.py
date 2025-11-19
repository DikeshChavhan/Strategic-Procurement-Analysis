# app.py — Final production-ready (Dark UI + Cloud Chatbot + Multi-language + Full Home)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Semantic chatbot (sentence-transformers)
from sentence_transformers import SentenceTransformer, util

# -------------------------
# App config + dark theme
# -------------------------
st.set_page_config(page_title="Kraljic Matrix Classifier — Pro", layout="wide")
dark_css = """
<style>
    .stApp { background-color: #0b1220; color: #e6eef8; }
    .stSidebar { background-color: #071022; color: #e6eef8; }
    .card { background-color: #0f1724; padding: 14px; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
    .muted { color: #9aa6b2; }
    .title { font-size: 20px; font-weight: 700; }
    .small-muted { color: #9aa6b2; font-size:12px; }
    .stDownloadButton>button { background-color: #06b6d4; color: white; }
    .stButton>button { background-color: #06b6d4; color: white; }
    a { color: #7dd3fc }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# -------------------------
# Confirmed model features (in exact order)
# -------------------------
MODEL_FEATURES = [
    "Lead_Time_Days",
    "Order_Volume_Units",
    "Cost_per_Unit",
    "Supply_Risk_Score",
    "Profit_Impact_Score",
    "Environmental_Impact",
    "Supplier_Region",
    "Single_Source_Risk"
]

# -------------------------
# Load model (must exist)
# -------------------------
MODEL_PATH = "naive_bayes_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Place it in the same folder as app.py and restart.")
    st.stop()

model = joblib.load(MODEL_PATH)

# If model provides feature_names_in_ prefer that (but we assume the confirmed order is correct)
model_cols = list(getattr(model, "feature_names_in_", MODEL_FEATURES))

# -------------------------
# Minimal translations
# -------------------------
LANG = {
    "English": {
        "title": "Kraljic Matrix Classifier — Pro",
        "home_desc": "Classify procurement items into Strategic / Leverage / Bottleneck / Non-Critical. Use single item mode or upload a batch CSV.",
        "predict": "Predict",
        "download_input": "Download input (CSV)",
        "download_preds": "Download predictions (CSV)",
        "about_title": "About this App",
        "about_text": """**What it does:** Predicts the Kraljic category for procurement items using a trained Naive Bayes model.  
**Why:** Helps procurement teams prioritize sourcing strategy and mitigate supply risk.  
**Use cases:** Manufacturing, retail, procurement teams, students.""",
        "chat_title": "AI Assistant",
        "chat_placeholder": "Ask procurement questions (e.g. 'What is a bottleneck item?')",
        "chat_ask": "Ask"
    },
    "Hindi": {
        "title": "Kraljic मैट्रिक्स क्लासिफायर — प्रो",
        "home_desc": "यह ऐप सामानों को Strategic / Leverage / Bottleneck / Non-Critical में वर्गीकृत करता है।",
        "predict": "पूर्वानुमान लगाएँ",
        "download_input": "इनपुट डाउनलोड करें (CSV)",
        "download_preds": "पूर्वानुमान डाउनलोड करें (CSV)",
        "about_title": "इस ऐप के बारे में",
        "about_text": """**क्या करता है:** Naive Bayes मॉडल का उपयोग करके Kraljic श्रेणी की भविष्यवाणी करता है।""",
        "chat_title": "एआई सहायक",
        "chat_placeholder": "प्रोक्योरमेंट से संबंधित प्रश्न पूछें (उदा. 'बॉटलनेक क्या है?')",
        "chat_ask": "पूछें"
    },
    "Marathi": {
        "title": "Kraljic मॅट्रिक्स क्लासिफायर — प्रो",
        "home_desc": "हा अॅप वस्तूंना Strategic / Leverage / Bottleneck / Non-Critical मध्ये वर्गीकृत करतो.",
        "predict": "भविष्यवाणी करा",
        "download_input": "इनपुट डाउनलोड (CSV)",
        "download_preds": "भविष्यवाणी डाउनलोड (CSV)",
        "about_title": "अॅपबद्दल",
        "about_text": """**हे काय करते:** Naive Bayes मॉडेल वापरून Kraljic वर्गीकरण करतो.""",
        "chat_title": "एआय सहाय्यक",
        "chat_placeholder": "प्रोक्योरमेंट विषयी प्रश्न विचारा (उदा. 'बॉटलनेक म्हणजे काय?')",
        "chat_ask": "वाचा"
    }
}

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Settings")
language = st.sidebar.selectbox("Language", list(LANG.keys()), index=0)
T = LANG[language]

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", ["Home", "Batch", "Chatbot", "About"])

# -------------------------
# Semantic Chatbot setup (cloud-friendly)
# -------------------------
# Knowledge base (small FAQ). You can extend these strings.
KB_PAIRS = [
    ("What is the Kraljic Matrix?", "The Kraljic Matrix classifies purchases into Strategic, Leverage, Bottleneck, and Non-Critical based on supply risk and profit impact."),
    ("What is a strategic item?", "Strategic items have high supply risk and high profit impact. Treat them with long-term partnerships and supplier development."),
    ("What is a bottleneck item?", "Bottleneck items have high supply risk and low profit impact. Diversify suppliers and maintain safety stock."),
    ("What is a leverage item?", "Leverage items have low supply risk and high profit impact. Use competitive bidding and negotiate better terms."),
    ("What is a non-critical item?", "Non-critical items have low supply risk and low profit impact. Automate procurement and focus on process efficiency.")
]

# load sentence-transformer embedding model (small and cloud-friendly)
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()
kb_texts = [q for q, a in KB_PAIRS]
kb_answers = [a for q, a in KB_PAIRS]
kb_embeddings = embed_model.encode(kb_texts, convert_to_tensor=True)

def kb_answer(user_q, k=2):
    # embed, compute similarity, return best answer (or fallback)
    q_emb = embed_model.encode(user_q, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, kb_embeddings)[0]
    # top k
    top_results = np.argpartition(-cos_scores.cpu().numpy(), range(min(k, len(kb_texts))))[:k]
    best_idx = int(top_results[0])
    score = float(cos_scores[best_idx])
    if score < 0.45:
        # low similarity -> fallback
        return None, score
    return kb_answers[best_idx], score

# -------------------------
# Helper functions: prepare input to model
# -------------------------
def prepare_input_row(df_row: pd.DataFrame):
    """
    df_row: dataframe with columns matching MODEL_FEATURES or user-provided names.
    - convert Single_Source_Risk (Yes/No) -> 1/0
    - keep Supplier_Region as-is (model trained on it)
    - reindex columns to model_cols order (fill missing with 0)
    """
    d = df_row.copy()
    if "Single_Source_Risk" in d.columns:
        d["Single_Source_Risk"] = d["Single_Source_Risk"].map({ "Yes": 1, "No": 0 }).fillna(d["Single_Source_Risk"])
    # Ensure Supplier_Region present; if not present fill with "Unknown"
    if "Supplier_Region" not in d.columns:
        d["Supplier_Region"] = "Unknown"
    # Reindex to model columns exactly
    final = d.reindex(columns=model_cols, fill_value=0)
    return final

def recommended_actions(cat):
    mapping = {
        "Strategic": ["Long-term partnerships", "Safety stock & contingency", "Supplier development"],
        "Leverage": ["Negotiate volume discounts", "Competitive bidding", "Consolidate spend"],
        "Bottleneck": ["Diversify suppliers", "Increase monitoring & safety stock", "Consider redesign"],
        "Non-Critical": ["Automate procurement", "Standardize items", "Optimize process costs"]
    }
    return mapping.get(cat, [])

# -------------------------
# PAGE: HOME (single item)
# -------------------------
if page == "Home":
    st.markdown(f"<div class='card'><div class='title'>{T['title']}</div><div class='small-muted'>{T['home_desc']}</div></div>", unsafe_allow_html=True)
    st.write("")  # spacer

    st.header("Single Item Prediction")
    cols = st.columns(2)
    with cols[0]:
        lead_time = st.number_input("Lead Time (Days)", min_value=0, max_value=3650, value=30)
        order_volume = st.number_input("Order Volume (Units)", min_value=1, value=500)
        cost_per_unit = st.number_input("Cost per Unit", min_value=0.01, value=250.0)
        supply_risk = st.slider("Supply Risk (1=Low → 5=High)", 1, 5, 3)
    with cols[1]:
        profit_impact = st.slider("Profit Impact (1=Low → 5=High)", 1, 5, 3)
        env_impact = st.slider("Environmental Impact (1=Low → 5=High)", 1, 5, 2)
        supplier_region = st.selectbox("Supplier Region", ["Maharashtra","Gujarat","Karnataka","Delhi NCR","Tamil Nadu","West Bengal","Rajasthan","Uttar Pradesh","Kerala","Punjab","China","Bangladesh","GCC","USA","Europe","Other"])
        single_source = st.selectbox("Single Source Risk?", ["Yes","No"])

    input_df = pd.DataFrame([{
        "Lead_Time_Days": lead_time,
        "Order_Volume_Units": order_volume,
        "Cost_per_Unit": cost_per_unit,
        "Supply_Risk_Score": supply_risk,
        "Profit_Impact_Score": profit_impact,
        "Environmental_Impact": env_impact,
        "Supplier_Region": supplier_region,
        "Single_Source_Risk": single_source
    }])

    st.subheader("Input Summary")
    st.table(input_df.T)

    if st.button(T["predict"] if "predict" in T else "Predict"):
        prepared = prepare_input_row(input_df)
        try:
            pred = model.predict(prepared)[0]
            st.success(f"🧩 Predicted Kraljic Category: **{pred}**")

            # show class probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(prepared)[0]
                st.subheader("Model confidence")
                st.bar_chart(pd.Series(probs, index=model.classes_))

            # recommendations
            st.subheader("Recommended actions")
            for r in recommended_actions(pred):
                st.write("•", r)

            # quadrant plot
            st.subheader("Kraljic Matrix Position")
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
            ax.set_xlabel("Profit Impact (1→5)")
            ax.set_ylabel("Supply Risk (1→5)")
            ax.axvline(3, color="white", linestyle="--"); ax.axhline(3, color="white", linestyle="--")
            ax.text(1,4.5,"Non-Critical", color="#a7f3d0")
            ax.text(3.2,4.5,"Leverage", color="#bfdbfe")
            ax.text(1,1,"Bottleneck", color="#fde68a")
            ax.text(3.2,1,"Strategic", color="#fecaca")
            ax.scatter(profit_impact, supply_risk, s=200, c="#06b6d4", marker="X")
            st.pyplot(fig)

        except Exception as e:
            st.error("Prediction error: " + str(e))

    # allow downloading the single input
    st.download_button(T["download_input"], input_df.to_csv(index=False).encode(), file_name="kraljic_input.csv", mime="text/csv")

# -------------------------
# PAGE: BATCH
# -------------------------
elif page == "Batch":
    st.header("Batch Predictions (CSV)")
    st.markdown("Upload CSV containing columns (names can include the model features):")
    st.code(", ".join(model_cols))
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded file:")
            st.dataframe(df.head())

            prepared = prepare_input_row(df)
            preds = model.predict(prepared)
            df["Predicted_Kraljic_Category"] = preds
            st.subheader("Results preview")
            st.dataframe(df.head())

            st.download_button(T["download_preds"], df.to_csv(index=False).encode(), file_name="kraljic_predictions.csv", mime="text/csv")
            st.subheader("Distribution")
            st.bar_chart(df["Predicted_Kraljic_Category"].value_counts())

        except Exception as e:
            st.error("Batch prediction error: " + str(e))

# -------------------------
# PAGE: CHATBOT
# -------------------------
elif page == "Chatbot":
    st.header(T["chat_title"])
    user_q = st.text_input(T["chat_placeholder"], "")
    if st.button(T["chat_ask"]):
        if not user_q.strip():
            st.warning("Please type a question.")
        else:
            ans, score = None, None
            try:
                ans, score = kb_answer(user_q, k=2)
            except Exception:
                ans = None
            if ans:
                st.success(f"Answer (similarity {score:.2f}):")
                st.write(ans)
            else:
                # fallback rule-based
                uq = user_q.lower()
                if "risk" in uq:
                    st.info("Risk increases with longer lead times, single-source dependency, and supplier instability.")
                elif "strategic" in uq:
                    st.info("Strategic = high risk + high profit. Use long-term supplier partnerships.")
                elif "bottleneck" in uq:
                    st.info("Bottleneck = high risk + low profit. Diversify suppliers & monitor closely.")
                else:
                    st.info("I couldn't find an exact match. Try asking about 'strategic', 'bottleneck', 'leverage', or 'risk'.")

# -------------------------
# PAGE: ABOUT
# -------------------------
elif page == "About":
    st.header(T["about_title"])
    st.markdown(T["about_text"])

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(f"Model expects features (in order): {', '.join(model_cols)} • Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
