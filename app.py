import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Kraljic Matrix Classifier", layout="wide")

LANGUAGES = ["English", "Hindi", "Marathi"]

# -------------------------------
# LANGUAGE TEXT
# -------------------------------
TEXT = {
    "English": {
        "title": "🧠 Kraljic Matrix Classifier — Smart Procurement Tool",
        "desc": "A practical procurement classification app using the **Kraljic Matrix**. Includes Indian supplier regions, batch CSV upload, charts, recommendations, and a built-in AI assistant.",
        "about_title": "ℹ️ About This App",
        "about_text": """
### What This App Does

This app helps procurement teams classify their items using the **Kraljic Matrix**, a strategic tool used worldwide.

### Why You Need It

✔ Identify high-risk or high-profit items  
✔ Prioritise suppliers  
✔ Improve sourcing strategies  
✔ Reduce supply chain risks  

### Where You Use It  
You can use this tool in:

- Manufacturing  
- Retail  
- Trading  
- Logistics  
- Indian supply chain operations  
- Vendor management  
""",
        "assistant_title": "🤖 AI Assistant",
        "assistant_placeholder": "Ask anything about procurement, supply chain, or the app…",
        "assistant_button": "Ask AI",
    },

    "Hindi": {
        "title": "🧠 क्रालजिक मैट्रिक्स क्लासिफायर — स्मार्ट प्रोक्योरमेंट टूल",
        "desc": "यह ऐप **Kraljic Matrix** का उपयोग करके खरीदारी वस्तुओं को श्रेणीबद्ध करता है। भारत-केन्द्रित सप्लायर क्षेत्र, CSV बैच अपलोड, चार्ट, सुझाव और एक AI सहायक शामिल है।",
        "about_title": "ℹ️ इस ऐप के बारे में",
        "about_text": """
### यह ऐप क्या करता है?

यह ऐप आपकी खरीदारी वस्तुओं को **Kraljic Matrix** के आधार पर वर्गीकृत करता है।

### यह क्यों जरूरी है?

✔ जोखिम वाले आइटम पहचानें  
✔ सप्लायर प्राथमिकता तय करें  
✔ बेहतर खरीद रणनीति बनाएं  
✔ सप्लाई चेन जोखिम कम करें  

### यह कहाँ उपयोग होता है?  
- मैन्युफैक्चरिंग  
- रिटेल  
- ट्रेडिंग  
- लॉजिस्टिक्स  
- भारतीय सप्लाई चेन  
""",
        "assistant_title": "🤖 एआई सहायक",
        "assistant_placeholder": "प्रोक्योरमेंट या इस ऐप के बारे में कुछ भी पूछें…",
        "assistant_button": "पूछें",
    },

    "Marathi": {
        "title": "🧠 क्रालजिक मॅट्रिक्स क्लासिफायर — स्मार्ट खरेदी साधन",
        "desc": "हे अ‍ॅप **Kraljic Matrix** वापरून खरेदीची वस्तू वर्गीकृत करते. भारतीय सप्लायर क्षेत्र, CSV अपलोड, चार्ट, शिफारसी आणि एआय सहाय्यक समाविष्ट.",
        "about_title": "ℹ️ अ‍ॅप बद्दल माहिती",
        "about_text": """
### हे अ‍ॅप काय करते?

हे अ‍ॅप तुमच्या खरेदी वस्तूंचे **Kraljic Matrix** वर आधारित वर्गीकरण करते.

### का वापरावे?

✔ जास्त जोखमीची वस्तू ओळखा  
✔ सप्लायरला प्राधान्य द्या  
✔ खरेदी धोरण सुधारवा  
✔ सप्लाय चेन रिस्क कमी करा  

### कुठे वापरू शकता?  
- मॅन्युफॅक्चरिंग  
- रिटेल  
- ट्रेडिंग  
- लॉजिस्टिक्स  
""",
        "assistant_title": "🤖 एआय सहाय्यक",
        "assistant_placeholder": "खरेदी किंवा अ‍ॅप बद्दल काहीही विचारा…",
        "assistant_button": "विचारा",
    }
}

# -------------------------------
# SIDEBAR OPTIONS
# -------------------------------
st.sidebar.title("Settings")

language = st.sidebar.selectbox("Choose Language", LANGUAGES)

page = st.sidebar.radio("Navigate", ["Home", "About", "AI Assistant"])

# -------------------------------
# LOAD MODEL
# -------------------------------
MODEL_PATH = "naive_bayes_model.pkl"
MODEL_COLUMNS_PATH = "model_columns.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)

default_columns = [
    "Lead_Time_Days",
    "Order_Volume_Units",
    "Cost_per_Unit",
    "Supply_Risk_Score",
    "Profit_Impact_Score",
    "Environmental_Impact",
    "Single_Source_Risk"
]

if os.path.exists(MODEL_COLUMNS_PATH):
    model_columns = list(joblib.load(MODEL_COLUMNS_PATH))
else:
    model_columns = default_columns

# -------------------------------
# FUNCTIONS
# -------------------------------
def prepare_input(df):
    if "Supplier_Region" in df.columns:
        df = df.drop(columns=["Supplier_Region"])

    df["Single_Source_Risk"] = df["Single_Source_Risk"].map({"Yes": 1, "No": 0})

    return df.reindex(columns=model_columns, fill_value=0)

def recommendations(cat):
    recs = {
        "Strategic": [
            "Develop long-term supplier partnerships.",
            "Joint forecasting & risk management.",
            "Supplier development programs."
        ],
        "Leverage": [
            "Competitive bidding.",
            "Volume consolidation.",
            "Aggressive negotiation."
        ],
        "Bottleneck": [
            "Identify backup suppliers.",
            "Increase safety stock.",
            "Explore material alternatives."
        ],
        "Non-Critical": [
            "Automate purchasing.",
            "Use long contracts.",
            "Focus on process efficiency."
        ]
    }
    return recs.get(cat, ["No recommendation."])

def chat_ai(message):
    # Simple rule-based AI (no external API required)
    if "risk" in message.lower():
        return "Risk depends on supplier reliability, lead time, and market volatility."
    if "kraljic" in message.lower():
        return "The Kraljic Matrix classifies items into: Strategic, Leverage, Bottleneck, and Non-critical."
    return "Thanks for your question! Based on procurement best practices, I recommend analysing supply risk and profit impact."

# -------------------------------
# PAGE: ABOUT
# -------------------------------
if page == "About":
    st.title(TEXT[language]["about_title"])
    st.markdown(TEXT[language]["about_text"])

# -------------------------------
# PAGE: AI ASSISTANT
# -------------------------------
elif page == "AI Assistant":
    st.title(TEXT[language]["assistant_title"])

    user_q = st.text_input(TEXT[language]["assistant_placeholder"])
    if st.button(TEXT[language]["assistant_button"]):
        if user_q.strip() == "":
            st.warning("Please enter a question.")
        else:
            answer = chat_ai(user_q)
            st.success(answer)

# -------------------------------
# PAGE: HOME (FULL APP)
# -------------------------------
else:
    st.title(TEXT[language]["title"])
    st.markdown(TEXT[language]["desc"])

    st.markdown("## 🔽 Prediction Options")
    mode = st.radio("Choose Mode", ["Single Item", "Batch CSV"])

    REGIONS = [
        "Maharashtra", "Gujarat", "Karnataka", "Delhi NCR", "Tamil Nadu",
        "West Bengal", "Rajasthan", "Uttar Pradesh", "Kerala", "Punjab",
        "China", "Bangladesh", "GCC", "USA", "Europe", "Other"
    ]

    # -----------------------
    # SINGLE ITEM MODE
    # -----------------------
    if mode == "Single Item":
        lead = st.number_input("Lead Time (Days)", 0, 3650, 30)
        vol = st.number_input("Order Volume (Units)", 1, 10_000_000, 500)
        cost = st.number_input("Cost per Unit", 1.0, 10_000_000.0, 250.0)
        risk = st.slider("Supply Risk", 1, 5, 3)
        impact = st.slider("Profit Impact", 1, 5, 3)
        env = st.slider("Environmental Impact", 1, 5, 2)
        ss = st.selectbox("Single Source Risk", ["Yes", "No"])
        region = st.selectbox("Supplier Region", REGIONS)

        df = pd.DataFrame({
            "Lead_Time_Days": [lead],
            "Order_Volume_Units": [vol],
            "Cost_per_Unit": [cost],
            "Supply_Risk_Score": [risk],
            "Profit_Impact_Score": [impact],
            "Environmental_Impact": [env],
            "Single_Source_Risk": [ss],
            "Supplier_Region": [region]
        })

        st.subheader("Input Summary")
        st.table(df.T)

        if st.button("Predict"):
            prepared = prepare_input(df)
            pred = model.predict(prepared)[0]
            proba = model.predict_proba(prepared)[0]

            st.success(f"Predicted Category: **{pred}**")

            st.subheader("Confidence")
            st.bar_chart(pd.Series(proba, index=model.classes_))

            st.subheader("Recommended Actions")
            for r in recommendations(pred):
                st.write("•", r)

            # Quadrant chart
            st.subheader("Kraljic Matrix Position")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
            ax.axvline(3, color="gray", linestyle="--")
            ax.axhline(3, color="gray", linestyle="--")
            ax.scatter(impact, risk, s=200, color="black")
            st.pyplot(fig)

    # -----------------------
    # BATCH MODE
    # -----------------------
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.dataframe(df.head())

            prepared = prepare_input(df)
            preds = model.predict(prepared)

            df["Predicted_Category"] = preds
            st.subheader("Results")
            st.dataframe(df)

            st.download_button("Download CSV", df.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

            st.subheader("Category Distribution")
            st.bar_chart(df["Predicted_Category"].value_counts())
