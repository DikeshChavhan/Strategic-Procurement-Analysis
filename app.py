# app.py — Upgraded: Dark theme, Professional UI, Local LLM assistant fallback, SQLite DB, Login/Admin
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import sqlite3
import hashlib
import time

# try to import local LLM (transformers). If not present, assistant will fallback to rule-based.
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -----------------------------
# CONFIG / THEME
# -----------------------------
st.set_page_config(page_title="Kraljic Procurement — Pro", layout="wide")

# Small dark theme via CSS (works without .streamlit config)
dark_css = """
<style>
    /* background */
    .stApp { background-color: #0f1724; color: #e6eef8; }
    .stButton>button { background-color: #0ea5a4; color: white; }
    .stSidebar { background-color: #071022; color: #e6eef8; }
    .card { background-color: #0b1220; padding: 16px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    .muted { color: #9aa6b2; }
    .title { font-size: 22px; font-weight: 600; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# -----------------------------
# DATABASE (SQLite)
# -----------------------------
DB_PATH = "app_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    # users table: username (unique), password_hash, is_admin
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )
    """)
    # predictions history
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        timestamp TEXT,
        input_json TEXT,
        predicted_category TEXT
    )
    """)
    conn.commit()
    return conn

conn = init_db()
cur = conn.cursor()

# create a default admin account if not exists (username: admin, password: admin123) — change ASAP
def create_default_admin():
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    if cur.fetchone() is None:
        pw_hash = hashlib.sha256("admin123".encode()).hexdigest()
        cur.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                    ("admin", pw_hash, 1))
        conn.commit()

create_default_admin()

# -----------------------------
# MODEL LOAD (compatible with model.feature_names_in_)
# -----------------------------
MODEL_PATH = "naive_bayes_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("Model file 'naive_bayes_model.pkl' not found. Place it in app folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
# prefer model.feature_names_in_ if available
model_columns = list(getattr(model, "feature_names_in_", [
    "Lead_Time_Days",
    "Order_Volume_Units",
    "Cost_per_Unit",
    "Supply_Risk_Score",
    "Profit_Impact_Score",
    "Environmental_Impact",
    "Single_Source_Risk",
    "Supplier_Region"
]))

# If the model expects numeric encoding for Supplier_Region, app will attempt to map common regions to ints
REGION_MAP = {
    "Maharashtra": 0, "Gujarat": 1, "Karnataka": 2, "Delhi NCR": 3, "Tamil Nadu": 4,
    "West Bengal": 5, "Rajasthan": 6, "Uttar Pradesh": 7, "Kerala": 8, "Punjab": 9,
    "China": 10, "Bangladesh": 11, "GCC": 12, "USA": 13, "Europe": 14, "Other": 15
}

# -----------------------------
# LOCAL LLM ASSISTANT SETUP (optional)
# -----------------------------
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", None)  # set env var to model path or HF id if available
llm_pipe = None
if TRANSFORMERS_AVAILABLE and LOCAL_LLM_MODEL:
    try:
        # load a text-generation / chat model pipeline if available
        llm_pipe = pipeline("text-generation", model=LOCAL_LLM_MODEL, tokenizer=LOCAL_LLM_MODEL, device=0)
    except Exception as e:
        llm_pipe = None

def llm_answer(question):
    # If we have a loaded pipeline, generate a short answer
    if llm_pipe is not None:
        try:
            out = llm_pipe(question, max_length=200, do_sample=False)
            return out[0]["generated_text"]
        except Exception:
            return None
    return None

# -----------------------------
# Auth helpers (simple)
# -----------------------------
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password, is_admin=0):
    pw_hash = hash_password(password)
    try:
        cur.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)", (username, pw_hash, is_admin))
        conn.commit()
        return True, "User registered."
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def login_user(username, password):
    pw_hash = hash_password(password)
    cur.execute("SELECT username, is_admin FROM users WHERE username = ? AND password_hash = ?", (username, pw_hash))
    row = cur.fetchone()
    if row:
        return True, {"username": row[0], "is_admin": bool(row[1])}
    return False, None

# -----------------------------
# UI Layout & Pages
# -----------------------------
st.sidebar.markdown("## 🔒 Account")
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "user": None}

if not st.session_state.auth["logged_in"]:
    action = st.sidebar.selectbox("Action", ["Login", "Register"])
    if action == "Login":
        uname = st.sidebar.text_input("Username")
        p = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            ok, info = login_user(uname, p)
            if ok:
                st.session_state.auth = {"logged_in": True, "user": info["username"], "is_admin": info["is_admin"]}
                st.sidebar.success("Logged in as " + info["username"])
            else:
                st.sidebar.error("Invalid credentials.")
    else:
        new_uname = st.sidebar.text_input("Choose username")
        new_pw = st.sidebar.text_input("Choose password", type="password")
        is_admin_flag = st.sidebar.checkbox("Create admin account (careful)", value=False)
        if st.sidebar.button("Register"):
            ok, msg = register_user(new_uname, new_pw, 1 if is_admin_flag else 0)
            if ok:
                st.sidebar.success(msg + " You can login now.")
            else:
                st.sidebar.error(msg)
else:
    st.sidebar.info(f"Signed in: {st.session_state.auth['user']}")
    if st.sidebar.button("Logout"):
        st.session_state.auth = {"logged_in": False, "user": None}
        st.experimental_rerun()

# navigation
page = st.sidebar.radio("Navigate", ["Home", "AI Assistant", "Admin"])

# -----------------------------
# Helper: prepare inputs to model
# -----------------------------
def prepare_input_df(df: pd.DataFrame):
    d = df.copy()
    # map Supplier_Region if model expects numeric
    if "Supplier_Region" in d.columns and "Supplier_Region" in model_columns:
        # if model expects numeric, replace strings with mapping ints
        if d["Supplier_Region"].dtype == object:
            d["Supplier_Region"] = d["Supplier_Region"].map(REGION_MAP).fillna(0)
    # map Single_Source_Risk yes/no to 1/0
    if "Single_Source_Risk" in d.columns:
        d["Single_Source_Risk"] = d["Single_Source_Risk"].map({"Yes": 1, "No": 0}).fillna(d["Single_Source_Risk"])
    # reindex to model columns (fill missing with 0)
    prepared = d.reindex(columns=model_columns, fill_value=0)
    return prepared

# -----------------------------
# PAGE: HOME (main app)
# -----------------------------
if page == "Home":
    st.markdown("<div class='card'><div class='title'>Kraljic Matrix — Smart Procurement</div><div class='muted'>Use this page to predict categories, visualize results and store history.</div></div>", unsafe_allow_html=True)
    st.write("")  # spacer

    cols = st.columns([2, 1])
    with cols[0]:
        st.header("Single Item Prediction")
        lead = st.number_input("Lead Time (Days)", min_value=0, max_value=3650, value=30)
        vol = st.number_input("Order Volume (Units)", min_value=1, value=500)
        cost = st.number_input("Cost per Unit", min_value=0.01, value=250.0)
        supply_risk = st.slider("Supply Risk Score (1-5)", 1, 5, 3)
        profit_impact = st.slider("Profit Impact Score (1-5)", 1, 5, 3)
        env_impact = st.slider("Environmental Impact (1-5)", 1, 5, 2)
        ss = st.selectbox("Single Source Risk", ["No", "Yes"])
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
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(prepared)[0]
                st.success(f"Predicted Category: {pred}")
                if proba is not None:
                    st.subheader("Confidence by class")
                    st.bar_chart(pd.Series(proba, index=model.classes_))

                # save to DB if logged in
                if st.session_state.auth["logged_in"]:
                    cur.execute("INSERT INTO predictions (username, timestamp, input_json, predicted_category) VALUES (?, ?, ?, ?)",
                                (st.session_state.auth["user"], time.ctime(), input_df.to_json(), str(pred)))
                    conn.commit()

                # recommendations simple mapping
                recs = {
                    "Strategic": ["Long term partnership", "Safety stock", "Supplier development"],
                    "Leverage": ["Negotiate bulk pricing", "Competitive bidding"],
                    "Bottleneck": ["Diversify sources", "Increase safety stock"],
                    "Non-Critical": ["Automate transactions", "Standardize SKUs"]
                }
                st.subheader("Recommended actions")
                for r in recs.get(pred, ["No specific actions"]):
                    st.write("•", r)

                # quadrant plot
                fig, ax = plt.subplots(figsize=(5,5))
                ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
                ax.axvline(3, color="gray", linestyle="--"); ax.axhline(3, color="gray", linestyle="--")
                ax.scatter(profit_impact, supply_risk, s=150, color="#00ffcc")
                ax.set_xlabel("Profit Impact"); ax.set_ylabel("Supply Risk")
                st.pyplot(fig)

            except Exception as e:
                st.error("Prediction error: " + str(e))

    with cols[1]:
        st.header("Batch Predictions")
        uploaded = st.file_uploader("Upload CSV with model features", type=["csv"])
        st.markdown("CSV must contain columns: " + ", ".join(model_columns))
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.write("Preview:")
                st.dataframe(df.head())
                prepared = prepare_input_df(df)
                preds = model.predict(prepared)
                df["Predicted_Category"] = preds
                st.dataframe(df.head())
                st.download_button("Download predictions CSV", df.to_csv(index=False).encode(), "preds.csv", "text/csv")
                # store first 100 rows in DB with username if logged in
                if st.session_state.auth["logged_in"]:
                    for _, row in df.head(100).iterrows():
                        cur.execute("INSERT INTO predictions (username, timestamp, input_json, predicted_category) VALUES (?, ?, ?, ?)",
                                    (st.session_state.auth["user"], time.ctime(), row.to_json(), str(row["Predicted_Category"])))
                    conn.commit()
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
        st.info("Login as admin to view saved prediction history.")

# -----------------------------
# PAGE: AI Assistant
# -----------------------------
elif page == "AI Assistant":
    st.markdown("<div class='card'><div class='title'>AI Assistant</div><div class='muted'>Ask procurement questions. If you added a local LLM, it will be used; otherwise a smart fallback is used.</div></div>", unsafe_allow_html=True)
    user_q = st.text_input("Ask a question about procurement or the app")
    if st.button("Ask"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            # try local LLM
            llm_resp = llm_answer(user_q) if TRANSFORMERS_AVAILABLE else None
            if llm_resp:
                st.write("**AI (local LLM):**")
                st.write(llm_resp)
            else:
                # rule-based fallback
                q = user_q.lower()
                if "risk" in q:
                    st.write("**Assistant:** Risk increases with longer lead times, single-source dependency, and market volatility.")
                elif "strategic" in q:
                    st.write("**Assistant:** Strategic items are high-impact and high-risk. Use long-term partnerships.")
                else:
                    st.write("**Assistant:** Try focusing on Supply Risk and Profit Impact to classify items. You can also upload a CSV to run batch classification.")

# -----------------------------
# PAGE: ADMIN (simple)
# -----------------------------
elif page == "Admin":
    st.header("Admin Panel")
    if not st.session_state.auth["logged_in"] or not st.session_state.auth.get("is_admin"):
        st.warning("Admin access only. Login as admin.")
    else:
        st.success(f"Welcome admin: {st.session_state.auth['user']}")
        # manage users
        st.subheader("Users")
        cur.execute("SELECT id, username, is_admin FROM users")
        users = cur.fetchall()
        u_df = pd.DataFrame(users, columns=["id","username","is_admin"])
        st.dataframe(u_df)
        # add new user
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

# -----------------------------
# CLEANUP on exit
# -----------------------------
# conn.close()  # keep DB open during app lifetime
