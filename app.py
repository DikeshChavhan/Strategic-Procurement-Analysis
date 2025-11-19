import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(page_title="Kraljic Procurement Classifier", layout="wide")

# --------------------------------------------------
# Language Strings
# --------------------------------------------------
LANG = {
    "English": {
        "title": "🧠 Kraljic Matrix Classification App",
        "about_title": "ℹ️ About This App",
        "about": """
### What this app does?

This app helps procurement professionals classify purchased materials into the **Kraljic Matrix**, 
which has 4 categories:

1. **Strategic** — High risk, high profit impact  
2. **Leverage** — Low risk, high profit impact  
3. **Bottleneck** — High risk, low profit impact  
4. **Non-Critical** — Low risk, low profit impact  

Companies use this classification to:
- Reduce supply risk  
- Improve supplier management  
- Decide negotiation strategies  
- Improve purchasing decisions  

This tool is useful for:
- Procurement teams  
- Supply chain students  
- Manufacturing businesses  
- Vendor management  
""",

        "chatbot_title": "🤖 Ask Your Doubts",
        "chat_placeholder": "Ask anything about procurement, Kraljic matrix, risk, etc...",
        "predict_button": "Predict Category",
        "download": "Download Input (CSV)",
    },

    "Hindi": {
        "title": "🧠 क्रैलजिक मैट्रिक्स वर्गीकरण ऐप",
        "about_title": "ℹ️ इस ऐप के बारे में",
        "about": """
### यह ऐप क्या करता है?

यह ऐप खरीद विभाग (Procurement) को यह समझने में मदद करता है कि  
कौन-सी खरीद सामग्री किस **Kraljic Matrix** श्रेणी में आती है:

1. **स्ट्रैटेजिक** — उच्च जोखिम, उच्च लाभ  
2. **लेवरेज** — कम जोखिम, उच्च लाभ  
3. **बॉटलनेक** — उच्च जोखिम, कम लाभ  
4. **नॉन-क्रिटिकल** — कम जोखिम, कम लाभ  

यह कंपनियों को मदद करता है:
- सप्लाई रिस्क कम करने में  
- बेहतर सप्लायर मैनेजमेंट में  
- नेगोशिएशन रणनीति तय करने में  
- सही खरीद निर्णय लेने में  
""",

        "chatbot_title": "🤖 अपने सवाल पूछें",
        "chat_placeholder": "प्रोक्योरमेंट या Kraljic मैट्रिक्स से संबंधित सवाल पूछें...",
        "predict_button": "श्रेणी बताएं",
        "download": "इनपुट डाउनलोड करें (CSV)",
    },

    "Marathi": {
        "title": "🧠 क्रॅलजिक मॅट्रिक्स वर्गीकरण अ‍ॅप",
        "about_title": "ℹ️ या अ‍ॅपबद्दल",
        "about": """
### हे अ‍ॅप काय करते?

हे अ‍ॅप खरेदी विभागाला (Procurement) मदत करते की  
सामग्री कोणत्या **Kraljic Matrix** श्रेणीत येते:

1. **Strategic** — जास्त रिस्क, जास्त नफा  
2. **Leverage** — कमी रिस्क, जास्त नफा  
3. **Bottleneck** — जास्त रिस्क, कमी नफा  
4. **Non-Critical** — कमी रिस्क, कमी नफा  

यामुळे कंपन्यांना फायदा:
- सप्लाय रिस्क कमी होतो  
- सप्लायर व्यवस्थापन सुधारते  
- नेगोशिएशन स्ट्रॅटेजी चांगली होते  
- खरेदी निर्णय सुधारतात  
""",

        "chatbot_title": "🤖 आपले प्रश्न विचारा",
        "chat_placeholder": "प्रोक्योरमेंट किंवा Kraljic Matrix बद्दल काहीही विचारा...",
        "predict_button": "श्रेणी दाखवा",
        "download": "इनपुट डाउनलोड (CSV)",
    }
}

# --------------------------------------------------
# Language Selector
# --------------------------------------------------
language = st.sidebar.selectbox("🌐 Choose Language / भाषा / भाषा निवडा", ["English", "Hindi", "Marathi"])
TXT = LANG[language]

st.title(TXT["title"])

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Enter Procurement Details")

lead_time = st.sidebar.number_input("Lead Time (Days)", min_value=1, max_value=365, value=30)
order_volume = st.sidebar.number_input("Order Volume (Units)", min_value=1, max_value=10000, value=500)
cost_per_unit = st.sidebar.number_input("Cost per Unit", min_value=0.1, max_value=10000.0, value=250.0)
supply_risk = st.sidebar.slider("Supply Risk Score (1-5)", 1, 5, 3)
profit_impact = st.sidebar.slider("Profit Impact Score (1-5)", 1, 5, 3)
env_impact = st.sidebar.slider("Environmental Impact (1-5)", 1, 5, 2)
region = st.sidebar.selectbox("Supplier Region", ["Asia", "Europe", "Africa", "North America", "South America"])
single_source = st.sidebar.selectbox("Single Source?", ["Yes", "No"])

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

# --------------------------------------------------
# Load Model
# --------------------------------------------------
try:
    model = joblib.load("naive_bayes_model.pkl")
except:
    st.error("Model file missing: naive_bayes_model.pkl")
    st.stop()

# --------------------------------------------------
# Tabs: About | Prediction | Chatbot
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📘 About", "📊 Predict", "🤖 Chatbot"])

# --------------------------------------------------
# TAB 1 – ABOUT SECTION
# --------------------------------------------------
with tab1:
    st.header(TXT["about_title"])
    st.write(TXT["about"])

# --------------------------------------------------
# TAB 2 – PREDICT
# --------------------------------------------------
with tab2:
    st.subheader("🔍 Input Summary")
    st.write(input_data)

    if st.button(TXT["predict_button"]):
        pred = model.predict(input_data)[0]
        st.success(f"### 🎯 Predicted Category: **{pred}**")

        csv = input_data.to_csv(index=False).encode()
        st.download_button(TXT["download"], csv, file_name="input.csv")

# --------------------------------------------------
# TAB 3 – CHATBOT
# --------------------------------------------------
with tab3:
    st.subheader(TXT["chatbot_title"])
    user_query = st.text_input(TXT["chat_placeholder"])

    if user_query:
        # Very basic chatbot response
        if "strategic" in user_query.lower():
            st.write("Strategic items = high risk + high impact. Used for critical suppliers.")
        elif "bottleneck" in user_query.lower():
            st.write("Bottleneck items = high risk + low impact. Need backup suppliers.")
        elif "leverage" in user_query.lower():
            st.write("Leverage items = low risk + high impact. Strong negotiation possible.")
        elif "non" in user_query.lower():
            st.write("Non-critical items = low cost, low risk, routine items.")
        else:
            st.write("This question seems related to procurement. Try asking about risk, suppliers, strategy, categories.")
