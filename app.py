# app.py — Final Production-ready (Dark UI, Hybrid Chatbot, SQLite Auth, Full Home)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import sqlite3
import hashlib
import time
import requests
import html
import difflib

# ---------------------------
# App config & dark theme CSS
# ---------------------------
st.set_page_config(page_title="Kraljic Procurement — Pro", layout="wide")

dark_css = """
<style>
:root{--bg:#071022;--card:#0b1220;--muted:#9aa6b2;--accent:#06b6d4;--text:#e6eef8}
.stApp { background: linear-gradient(180deg,#05101a 0%, #071427 100%); color: var(--text); }
.stSidebar { background: #071022; color: var(--text); }
.card{ background: var(--card); padding:16px; border-radius:12px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }
.title { font-size:20px; font-weight:700; color:var(--text); }
.muted { color: var(--muted); font-size:13px; }
.section { padding:12px 6px; }
.stButton>button { background: linear-gradient(90deg,var(--accent), #0ea5a4); color:white; border-radius:8px; }
.stDownloadButton>button { background: #10b981; color:white; border-radius:8px; }
.footer { color: var(--muted); font-size:12px; padding-top:8px; }
.chat-user { text-align:right; margin:6px 0; }
.chat-user .bubble { background:#064e3b; color:#e6fff6; padding:10px; display:inline-block; border-radius:10px; }
.chat-bot { text-align:left; margin:6px 0; }
.chat-bot .bubble { background:#0e1724; color:var(--text); padding:10px; display:inline-block; border-radius:10px; border:1px solid rgba(255,255,255,0.04); }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ---------------------------
# DB Init (SQLite)
# ---------------------------
DB_PATH = "app_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        timestamp TEXT,
        input_json TEXT,
        predicted_category TEXT
    )""")
    conn.commit()
    return conn

conn = init_db()
cur = conn.cursor()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def create_default_admin():
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                    ("admin", hash_password("admin123"), 1))
        conn.commit()

create_default_admin()

def register_user(username, password, is_admin=0):
    try:
        cur.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                    (username, hash_password(password), is_admin))
        conn.commit()
        return True, "User registered."
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def login_user(username, password):
    cur.execute("SELECT username, is_admin FROM users WHERE username = ? AND password_hash = ?",
                (username, hash_password(password)))
    row = cur.fetchone()
    if row:
        return True, {"username": row[0], "is_admin": bool(row[1])}
    return False, None

# ---------------------------
# Load model (confirmed feature order)
# ---------------------------
MODEL_PATH = "naive_bayes_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Add it to the app folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Confirmed feature order (user provided)
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

model_cols = list(getattr(model, "feature_names_in_", MODEL_FEATURES))

# region mapping (if numeric needed)
REGION_MAP = {
    "Maharashtra": 0, "Gujarat": 1, "Karnataka": 2, "Delhi NCR": 3, "Tamil Nadu": 4,
    "West Bengal": 5, "Rajasthan": 6, "Uttar Pradesh": 7, "Kerala": 8, "Punjab": 9,
    "China": 10, "Bangladesh": 11, "GCC": 12, "USA": 13, "Europe": 14, "Other": 15
}

# ---------------------------
# Sidebar: Auth & Nav
# ---------------------------
st.sidebar.markdown("## 🔒 Account")
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "user": None, "is_admin": False}

if not st.session_state.auth["logged_in"]:
    action = st.sidebar.selectbox("Action", ["Login", "Register"])
    if action == "Login":
        uname = st.sidebar.text_input("Username")
        pw = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            ok, info = login_user(uname, pw)
            if ok:
                st.session_state.auth = {"logged_in": True, "user": info["username"], "is_admin": info["is_admin"]}
                st.sidebar.success("Logged in as " + info["username"])
            else:
                st.sidebar.error("Invalid credentials.")
    else:
        new_uname = st.sidebar.text_input("Choose username")
        new_pw = st.sidebar.text_input("Choose password", type="password")
        is_admin_flag = st.sidebar.checkbox("Create admin account (careful)")
        if st.sidebar.button("Register"):
            ok, msg = register_user(new_uname, new_pw, 1 if is_admin_flag else 0)
            if ok:
                st.sidebar.success(msg + " You can login now.")
            else:
                st.sidebar.error(msg)
else:
    st.sidebar.info(f"Signed in: {st.session_state.auth['user']}")
    if st.sidebar.button("Logout"):
        st.session_state.auth = {"logged_in": False, "user": None, "is_admin": False}
        st.experimental_rerun()

page = st.sidebar.radio("Navigate", ["Home", "Batch", "Chatbot", "About", "Admin"])

# ---------------------------
# Helpers: prepare input
# ---------------------------
def prepare_input_df(df: pd.DataFrame):
    d = df.copy()
    # Supplier_Region mapping if model expects numeric
    if "Supplier_Region" in d.columns and "Supplier_Region" in model_cols:
        if d["Supplier_Region"].dtype == object:
            d["Supplier_Region"] = d["Supplier_Region"].map(REGION_MAP).fillna(0)
    if "Single_Source_Risk" in d.columns:
        d["Single_Source_Risk"] = d["Single_Source_Risk"].map({"Yes": 1, "No": 0}).fillna(d["Single_Source_Risk"])
    prepared = d.reindex(columns=model_cols, fill_value=0)
    return prepared

def recommended_actions(cat):
    recs = {
        "Strategic": ["Long-term partnerships", "Safety stock & contingency", "Supplier development"],
        "Leverage": ["Negotiate bulk pricing", "Competitive bidding", "Consolidate spend"],
        "Bottleneck": ["Diversify suppliers", "Increase monitoring & safety stock", "Explore alternatives"],
        "Non-Critical": ["Automate procurement", "Standardize items", "Focus on efficiency"]
    }
    return recs.get(cat, [])

# ---------------------------
# Hybrid Chatbot (OpenAI if key present, else fallback)
# ---------------------------
# FAQ knowledge base for fallback (extendable)
FAQ_PAIRS = [
    ("what is the kraljic matrix", "The Kraljic Matrix classifies purchases into Strategic, Leverage, Bottleneck and Non-Critical based on supply risk and profit impact."),
    ("what is a strategic item", "Strategic items have high supply risk and high profit impact. Use long-term partnerships and supplier development."),
    ("what is a bottleneck item", "Bottleneck items have high supply risk and low profit impact. Diversify suppliers and maintain safety stock."),
    ("what is a leverage item", "Leverage items have low supply risk and high profit impact. Use competitive bidding to reduce costs."),
    ("what is non-critical", "Non-critical items are low risk and low impact. Automate and standardize these purchases.")
]
FAQ_KEYS = [q for q,a in FAQ_PAIRS]
FAQ_ANSWERS = [a for q,a in FAQ_PAIRS]

def find_faq_answer(user_q, cutoff=0.45):
    q = user_q.lower().strip()
    for i, key in enumerate(FAQ_KEYS):
        if key in q:
            return FAQ_ANSWERS[i]
    # fuzzy match
    matches = difflib.get_close_matches(q, FAQ_KEYS, n=1, cutoff=cutoff)
    if matches:
        return FAQ_ANSWERS[FAQ_KEYS.index(matches[0])]
    return None

# Check for OpenAI key in Streamlit secrets or env
OPENAI_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
elif os.environ.get("OPENAI_API_KEY"):
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

def query_openai_chat(user_prompt, model_name="gpt-3.5-turbo", max_tokens=400):
    if not OPENAI_KEY:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role":"user","content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return None
    except Exception:
        return None

# Chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)

# ---------------------------
# PAGE: HOME (single item + UI)
# ---------------------------
if page == "Home":
    st.markdown("<div class='card'><div class='title'>Kraljic Matrix — Smart Procurement</div><div class='muted'>Predict categories, visualize, and optionally save history (login required).</div></div>", unsafe_allow_html=True)
    st.write("")
    cols = st.columns([2,1])
    with cols[0]:
        st.header("Single Item Prediction")
        lead = st.number_input("Lead Time (Days)", min_value=0, max_value=3650, value=30)
        vol = st.number_input("Order Volume (Units)", min_value=1, value=500)
        cost = st.number_input("Cost per Unit", min_value=0.01, value=250.0)
        supply_risk = st.slider("Supply Risk Score (1-5)", 1, 5, 3)
        profit_impact = st.slider("Profit Impact Score (1-5)", 1, 5, 3)
        env_impact = st.slider("Environmental Impact (1-5)", 1, 5, 2)
        ss = st.selectbox("Single Source Risk", ["No","Yes"])
        region = st.selectbox("Supplier Region", list(REGION_MAP.keys()))

        if st.button("Predict"):
            input_df = pd.DataFrame([{
                "Lead_Time_Days": lead,
                "Order_Volume_Units": vol,
                "Cost_per_Unit": cost,
                "Supply_Risk_Score": supply_risk,
                "Profit_Impact_Score": profit_impact,
                "Environmental_Impact": env_impact,
                "Single_Source_Risk": ss,
                "Supplier_Region": region
            }])
            prepared = prepare_input_df(input_df)
            try:
                pred = model.predict(prepared)[0]
                st.success(f"Predicted Category: {pred}")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(prepared)[0]
                    st.subheader("Confidence by class")
                    st.bar_chart(pd.Series(proba, index=model.classes_))
                # recommendations
                st.subheader("Recommended actions")
                for r in recommended_actions(pred):
                    st.write("•", r)
                # quadrant plot
                fig, ax = plt.subplots(figsize=(5,5))
                ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
                ax.axvline(3, color="gray", linestyle="--"); ax.axhline(3, color="gray", linestyle="--")
                ax.scatter(profit_impact, supply_risk, s=150, color="#06b6d4")
                ax.set_xlabel("Profit Impact"); ax.set_ylabel("Supply Risk")
                st.pyplot(fig)
                # save
                if st.session_state.auth["logged_in"]:
                    cur.execute("INSERT INTO predictions (username, timestamp, input_json, predicted_category) VALUES (?, ?, ?, ?)",
                                (st.session_state.auth["user"], time.ctime(), input_df.to_json(), str(pred)))
                    conn.commit()
            except Exception as e:
                st.error("Prediction error: " + str(e))
    with cols[1]:
        st.header("Batch Predictions")
        uploaded = st.file_uploader("Upload CSV with model features", type=["csv"])
        st.markdown("CSV should contain columns (names matching or including): " + ", ".join(model_cols))
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.write("Preview:")
                st.dataframe(df.head())
                prepared = prepare_input_df(df)
                preds = model.predict(prepared)
                df["Predicted_Category"] = preds
                st.dataframe(df.head())
                st.download_button("Download predictions (CSV)", df.to_csv(index=False).encode(), "preds.csv", "text/csv")
            except Exception as e:
                st.error("Batch processing error: " + str(e))
    st.markdown("---")
    st.subheader("Recent Predictions (admin only)")
    if st.session_state.auth.get("is_admin"):
        cur.execute("SELECT id, username, timestamp, predicted_category FROM predictions ORDER BY id DESC LIMIT 50")
        rows = cur.fetchall()
        if rows:
            recent_df = pd.DataFrame(rows, columns=["id","username","timestamp","predicted_category"])
            st.dataframe(recent_df)
        else:
            st.info("No saved predictions yet.")
    else:
        st.info("Login as admin to view saved history.")

# ---------------------------
# PAGE: Chatbot (Hybrid)
# ---------------------------
elif page == "Chatbot":
    st.markdown("<div class='card'><div class='title'>AI Assistant</div><div class='muted'>Ask procurement questions — uses cloud AI if API key present, otherwise a smart fallback.</div></div>", unsafe_allow_html=True)
    st.write("")
    # show chat history
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-user'><div class='bubble'>{html.escape(text)}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'><div class='bubble'>{html.escape(text)}</div></div>", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=False):
        user_input = st.text_input("Type your question...", key="chat_input")
        submit = st.form_submit_button("Ask")
        if submit:
            user_input = user_input.strip()
            if not user_input:
                st.warning("Please enter a question.")
            else:
                # append user
                st.session_state.chat_history.append(("user", user_input))

                # 1) Try OpenAI if key available
                answer = None
                if OPENAI_KEY:
                    try:
                        # small prompt: keep answers focused and short for procurement context
                        prompt = f"You are an expert procurement assistant. Answer concisely: {user_input}"
                        ai_resp = query_openai_chat(prompt)
                        if ai_resp:
                            answer = ai_resp
                    except Exception:
                        answer = None

                # 2) Try FAQ fuzzy match fallback
                if not answer:
                    answer = find_faq_answer(user_input, cutoff=0.45)

                # 3) Rule-based fallback if still None
                if not answer:
                    q = user_input.lower()
                    if "risk" in q:
                        answer = "Risk increases with longer lead times, limited suppliers, and market volatility. Diversify suppliers, use safety stock, and create contingency plans."
                    elif "strategic" in q:
                        answer = "Strategic items: high risk & high impact — use long-term partnerships, tight collaboration and supplier development."
                    elif "bottleneck" in q:
                        answer = "Bottleneck items: high risk, low impact — find alternate sources, increase monitoring, and maintain safety stock."
                    else:
                        answer = ("I couldn't find an exact match. Try asking: 'what is strategic', 'what is bottleneck', 'how to reduce risk', "
                                  "or upload a CSV for batch predictions.")

                # append assistant
                st.session_state.chat_history.append(("assistant", answer))
                st.experimental_rerun()

# ---------------------------
# PAGE: About
# ---------------------------
elif page == "About":
    st.markdown("<div class='card'><div class='title'>About this App</div><div class='muted'>Read why and how to use it</div></div>", unsafe_allow_html=True)
    st.markdown("""
**What it does:** Predicts Kraljic categories (Strategic, Leverage, Bottleneck, Non-Critical) using a saved Naive Bayes model.

**Why:** Helps procurement teams prioritize supplier strategy, manage risk, and focus on value.

**How to use:** Enter item attributes in *Home*, or upload a CSV in *Batch*. Use *Chatbot* for questions.

**Admin:** Use default admin (admin / admin123) to see saved history; change default credentials after first login.
""")

# ---------------------------
# PAGE: Admin
# ---------------------------
elif page == "Admin":
    st.header("Admin")
    if not st.session_state.auth["logged_in"] or not st.session_state.auth.get("is_admin"):
        st.warning("Admin access only. Login as admin.")
    else:
        st.success(f"Welcome admin: {st.session_state.auth['user']}")
        st.subheader("Users")
        cur.execute("SELECT id, username, is_admin FROM users")
        users = cur.fetchall()
        u_df = pd.DataFrame(users, columns=["id","username","is_admin"])
        st.dataframe(u_df)
        st.subheader("Create user")
        new_user = st.text_input("Username for new user")
        new_pw = st.text_input("Password for new user", type="password")
        make_admin = st.checkbox("Make admin")
        if st.button("Create user"):
            ok, msg = register_user(new_user, new_pw, 1 if make_admin else 0)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(f"Model expects features (in order): {', '.join(model_cols)} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
