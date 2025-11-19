import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# =========================================
# APP CONFIG
# =========================================
st.set_page_config(page_title="Kraljic Matrix Classifier", layout="wide")

# ============================
# LANGUAGE PACKS
# ============================
LANG = {
    "English": {
        "title": "🧠 Kraljic Matrix Classification — AI Powered",
        "about_title": "ℹ️ About This App",
        "about_text": """
### 🌐 What is this app?

This is an AI-powered **Kraljic Matrix Classification Tool** that helps procurement
teams and businesses understand the risk & impact of purchased materials.

### 🧩 What is Kraljic Matrix?

The Kraljic Matrix classifies purchased items into four categories:

- **Strategic** (High Risk, High Impact)  
- **Leverage** (Low Risk, High Impact)  
- **Bottleneck** (High Risk, Low Impact)  
- **Non-Critical** (Low Risk, Low Impact)

### 🎯 How this app helps?

This tool:
- Predicts the item category using your trained Naive Bayes Model  
- Shows charts & quadrant visualization  
- Provides procurement recommendations  
- Supports **single item** & **batch CSV upload**  
- Supports **English, Hindi, Marathi**  
- Includes a built-in **AI Assistant**  

### 🇮🇳 India-focused suppliers included:
Maharashtra, Gujarat, Karnataka, Tamil Nadu, Delhi NCR, UP, Rajasthan, Punjab, Kerala etc.

### 👤 Who should use this app?
- Supply Chain Students  
- Procurement Managers  
- Manufacturing Units  
- Researchers  
""",
        "assistant_title": "🤖 AI Procurement Assistant",
        "assistant_placeholder": "Ask any procurement question..."
    },

    "Hindi": {
        "title": "🧠 क्रैलजिक मैट्रिक्स वर्गीकरण — एआई आधारित",
        "about_title": "ℹ️ इस ऐप के बारे में",
        "about_text": """
### 🌐 यह ऐप क्या करता है?

यह एआई आधारित **Kraljic Matrix Classification Tool** खरीद (Procurement) में  
जोखिम और प्रभाव का विश्लेषण करता है।

### 🧩 क्रैलजिक मैट्रिक्स क्या है?

यह किसी भी खरीदे गए आइटम को चार श्रेणियों में बांटता है:

- **Strategic** (उच्च जोखिम • उच्च प्रभाव)  
- **Leverage** (कम जोखिम • उच्च प्रभाव)  
- **Bottleneck** (उच्च जोखिम • कम प्रभाव)  
- **Non-Critical** (कम जोखिम • कम प्रभाव)

### 🎯 यह ऐप आपकी कैसे मदद करेगा?

- AI मॉडल से सही वर्गीकरण  
- चार्ट, विज़ुअल, रिकमेन्डेशन  
- एकल या CSV बैच अपलोड  
- **हिंदी, अंग्रेजी, मराठी** सपोर्ट  
- बिल्ट-इन **एआई असिस्टेंट**  

### 🇮🇳 भारत आधारित सप्लायर रीजन:
महाराष्ट्र, गुजरात, कर्नाटक, दिल्ली NCR, तमिलनाडु आदि।

### 👤 कौन उपयोग कर सकता है?
- सप्लाई चेन छात्र  
- प्रोक्योरमेंट मैनेजर  
- मैन्युफैक्चरिंग यूनिट्स  
""",
        "assistant_title": "🤖 एआई प्रोक्योरमेंट असिस्टेंट",
        "assistant_placeholder": "अपना सवाल पूछें..."
    },

    "Marathi": {
        "title": "🧠 क्रॅलजिक मॅट्रिक्स वर्गीकरण — एआय आधारित",
        "about_title": "ℹ️ अॅप बद्दल माहिती",
        "about_text": """
### 🌐 हे अॅप काय करतो?

हे एआय-आधारित **Kraljic Matrix Classification Tool** खरेदीत  
जोखीम आणि प्रभाव समजण्यासाठी मदत करते.

### 🧩 क्रॅलजिक मॅट्रिक्स काय आहे?

खरेदी केलेल्या वस्तू खालील 4 वर्गात मोडतात:

- **Strategic** (जास्त जोखीम • जास्त प्रभाव)  
- **Leverage** (कमी जोखीम • जास्त प्रभाव)  
- **Bottleneck** (जास्त जोखीम • कमी प्रभाव)  
- **Non-Critical** (कमी जोखीम • कमी प्रभाव)

### 🎯 या अॅपचे फायदे:

- एआय मॉडेलवर आधारित अचूक भविष्यवाणी  
- चार्ट, क्वाड्रंट, रिकमेन्डेशन्स  
- एकल व CSV बॅच  
- **मराठी, हिंदी, इंग्रजी** भाषा  
- बिल्ट-इन **एआय असिस्टंट**  

### 🇮🇳 भारतातील पुरवठादार प्रदेश:
महाराष्ट्र, गुजरात, कर्नाटक, दिल्ली NCR, तामिळनाडू इत्यादी.

### 👤 कोण वापरू शकतो?
- सप्लाय चेन विद्यार्थी  
- प्रोक्योरमेंट मॅनेजर  
- उद्योग  
""",
        "assistant_title": "🤖 एआय प्रोक्योरमेंट सहाय्यक",
        "assistant_placeholder": "प्रश्न विचारा..."
    }
}

# =========================================
# SIDEBAR — LANGUAGE + NAVIGATION
# =========================================
st.sidebar.title("🌐 Language / भाषा / भाषा")
language = st.sidebar.selectbox("Select Language", ["English", "Hindi", "Marathi"])
T = LANG[language]

page = st.sidebar.radio("Navigate", ["Home", "AI Assistant", "About App"])


# =========================================
# LOAD YOUR MODEL
# =========================================
MODEL_PATH = "naive_bayes_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file missing!")
    st.stop()

model = joblib.load(MODEL_PATH)

model_columns = [
    "Lead_Time_Days",
    "Order_Volume_Units",
    "Cost_per_Unit",
    "Supply_Risk_Score",
    "Profit_Impact_Score",
    "Environmental_Impact",
    "Single_Source_Risk"
]

REGIONS = [
    "Maharashtra", "Gujarat", "Karnataka", "Delhi NCR", "Tamil Nadu",
    "West Bengal", "Rajasthan", "Uttar Pradesh", "Kerala", "Punjab",
    "China", "Bangladesh", "GCC", "USA", "Europe", "Other"
]

# =========================================
# HELPER FUNCTIONS
# =========================================
def prepare_input(df):
    df = df.copy()
    if "Single_Source_Risk" in df:
        df["Single_Source_Risk"] = df["Single_Source_Risk"].map({"Yes": 1, "No": 0}).fillna(df["Single_Source_Risk"])
    return df[model_columns]


def assistant_reply(q):
    q = q.lower()

    if "supplier" in q:
        return "A good supplier should have low risk, good lead time, and stable pricing."
    if "risk" in q:
        return "Risk increases with higher lead time, poor reliability, or single-source dependency."
    if "strategic" in q:
        return "Strategic items need long-term relations and strong collaboration."
    if "hello" in q or "hi" in q:
        return "Hello! How can I help you with procurement today?"

    return "I am not fully sure, but this seems related to procurement or supply chain."

# =========================================
# PAGE 1 — HOME (Your original prediction UI)
# =========================================
if page == "Home":
    st.title(T["title"])

    # Your entire original Single Item + Batch UI will be placed here
    # (I can merge it for you exactly once you confirm structure)

    st.info("Your home prediction page content remains here. (Same as your original code)")

# =========================================
# PAGE 2 — AI ASSISTANT
# =========================================
elif page == "AI Assistant":
    st.title(T["assistant_title"])

    user_q = st.text_input(T["assistant_placeholder"])

    if user_q:
        st.write("**You:**", user_q)
        st.write("**Assistant:**", assistant_reply(user_q))

# =========================================
# PAGE 3 — ABOUT APP
# =========================================
elif page == "About App":
    st.title(T["about_title"])
    st.markdown(T["about_text"])
