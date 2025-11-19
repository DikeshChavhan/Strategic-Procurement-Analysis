import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
model = joblib.load("naive_bayes_model.pkl")

FEATURES = [
    "Lead_Time_Days",
    "Order_Volume_Units",
    "Cost_per_Unit",
    "Supply_Risk_Score",
    "Profit_Impact_Score",
    "Environmental_Impact",
    "Supplier_Region",
    "Single_Source_Risk"
]

# -----------------------------------------------------
# Language Packs
# -----------------------------------------------------
LANG = {
    "English": {
        "title": "🌐 Strategic Procurement Risk Analyzer",
        "sidebar_title": "Navigation",
        "input_section": "📊 Enter Supplier Data",
        "predict_button": "Run Prediction",
        "result_title": "🔍 Prediction Result",
        "chat_title": "💬 Ask Procurement Chatbot",
        "about_title": "ℹ️ About this App",
        "about_text": """
This AI-powered tool predicts procurement risk levels using 
Naive Bayes classification. It supports multilingual UI and includes 
a built-in chatbot for procurement-related queries.
        """,
        "chat_placeholder": "Type your question..."
    },
    "Hindi": {
        "title": "🌐 रणनीतिक खरीद जोखिम विश्लेषक",
        "sidebar_title": "नेविगेशन",
        "input_section": "📊 सप्लायर डेटा दर्ज करें",
        "predict_button": "पूर्वानुमान चलाएँ",
        "result_title": "🔍 परिणाम",
        "chat_title": "💬 खरीद चैटबॉट से पूछें",
        "about_title": "ℹ️ ऐप के बारे में",
        "about_text": """
यह AI-आधारित टूल Naive Bayes मॉडल का उपयोग करके खरीद जोखिम की 
भविष्यवाणी करता है। मल्टी-लैंग्वेज सपोर्ट और बिल्ट-इन चैटबॉट मौजूद है।
        """,
        "chat_placeholder": "अपना सवाल लिखें..."
    },
    "Marathi": {
        "title": "🌐 धोरणात्मक खरेदी जोखीम विश्लेषक",
        "sidebar_title": "नेव्हिगेशन",
        "input_section": "📊 पुरवठादार माहिती भरा",
        "predict_button": "भविष्यवाणी चालवा",
        "result_title": "🔍 परिणाम",
        "chat_title": "💬 खरेदी चैटबॉटला विचारा",
        "about_title": "ℹ️ अॅप बद्दल माहिती",
        "about_text": """
हा AI टूल Naive Bayes मॉडेल वापरून खरेदी जोखीम स्तराचा अंदाज लावतो.
मल्टी-लँग्वेज सपोर्ट आणि सोपा चैटबॉट देखील उपलब्ध आहे.
        """,
        "chat_placeholder": "आपला प्रश्न टाइप करा..."
    }
}

# -----------------------------------------------------
# Simple Chatbot Logic
# -----------------------------------------------------
def chatbot_response(q):
    q = q.lower()

    if "risk" in q:
        return "Supplier risk depends on lead time, region, and single-source dependency."
    if "best supplier" in q:
        return "Best suppliers have low risk, high reliability, and stable pricing."
    if "cost" in q:
        return "Cost impact increases with high order volume or unstable pricing."
    if "hello" in q or "hi" in q:
        return "Hello! How can I assist in procurement analysis today?"
    return "I’m not fully sure, but this relates to procurement strategy or supplier management."

# -----------------------------------------------------
# Streamlit App
# -----------------------------------------------------
st.set_page_config(page_title="Procurement Analyzer", layout="wide")

# Language Selector
language = st.sidebar.selectbox("🌐 Choose Language", ["English", "Hindi", "Marathi"])
T = LANG[language]

st.title(T["title"])

# Sidebar Navigation
page = st.sidebar.radio(
    T["sidebar_title"],
    ["Home", "Chatbot", "About"]
)

# -----------------------------------------------------
# HOME PAGE – Prediction UI
# -----------------------------------------------------
if page == "Home":
    st.header(T["input_section"])
    
    lead_time = st.number_input("Lead Time (Days)", min_value=1, max_value=365, value=30)
    order_volume = st.number_input("Order Volume (Units)", min_value=1, value=100)
    cost_per_unit = st.number_input("Cost per Unit", min_value=0.1, value=10.0)
    supply_risk = st.slider("Supply Risk Score", 1, 10, 5)
    profit_impact = st.slider("Profit Impact Score", 1, 10, 6)
    env_impact = st.slider("Environmental Impact", 1, 10, 5)
    region = st.selectbox("Supplier Region", ["North", "South", "East", "West"])
    single_source = st.selectbox("Single Source Risk", [0, 1])

    region_map = {"North": 0, "South": 1, "East": 2, "West": 3}

    if st.button(T["predict_button"]):
        input_data = pd.DataFrame([{
            "Lead_Time_Days": lead_time,
            "Order_Volume_Units": order_volume,
            "Cost_per_Unit": cost_per_unit,
            "Supply_Risk_Score": supply_risk,
            "Profit_Impact_Score": profit_impact,
            "Environmental_Impact": env_impact,
            "Supplier_Region": region_map[region],
            "Single_Source_Risk": single_source
        }])

        pred = model.predict(input_data)[0]
        st.success(f"{T['result_title']}: **{pred}**")

# -----------------------------------------------------
# CHATBOT PAGE
# -----------------------------------------------------
elif page == "Chatbot":
    st.header(T["chat_title"])
    user_q = st.text_input(T["chat_placeholder"])

    if user_q:
        st.write("**You:**", user_q)
        st.write("**Bot:**", chatbot_response(user_q))

# -----------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------
elif page == "About":
    st.header(T["about_title"])
    st.write(T["about_text"])
