import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import httpx
import hashlib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Supabase config ───────────────────────────────────────
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
HEADERS = {
    "apikey"       : SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type" : "application/json"
}

# ── Email config ──────────────────────────────────────────
RESEND_API_KEY = st.secrets["RESEND_API_KEY"]
ADMIN_EMAIL    = st.secrets["ADMIN_EMAIL"]

# ── Email function ────────────────────────────────────────
def send_email(to, subject, html):
    httpx.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type" : "application/json"
        },
        json={
            "from"   : "CareSignal <onboarding@resend.dev>",
            "to"     : [to],
            "subject": subject,
            "html"   : html
        }
    )

def notify_admin(full_name, email, hospital, job_title, reason):
    send_email(
        to      = ADMIN_EMAIL,
        subject = "New CareSignal Access Request",
        html    = f"""
        <h2>New registration received</h2>
        <p><b>Name:</b> {full_name}</p>
        <p><b>Email:</b> {email}</p>
        <p><b>Hospital:</b> {hospital}</p>
        <p><b>Job title:</b> {job_title}</p>
        <p><b>Reason:</b> {reason}</p>
        <br>
        <p>Log in to Supabase to approve this user.</p>
        """
    )

# ── Auth helper functions ─────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(full_name, email, password, hospital, job_title, reason):
    data = {
        "full_name"    : full_name,
        "email"        : email,
        "password"     : hash_password(password),
        "hospital_name": hospital,
        "job_title"    : job_title,
        "reason"       : reason,
        "status"       : "pending"
    }
    r = httpx.post(f"{SUPABASE_URL}/rest/v1/users", json=data, headers=HEADERS)
    return r.status_code == 201

def login_user(email, password):
    hashed = hash_password(password)
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/users",
        params={"email": f"eq.{email}", "password": f"eq.{hashed}"},
        headers=HEADERS
    )
    if r.status_code == 200 and len(r.json()) > 0:
        return r.json()[0]
    return None

# ── Model ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Outputs') + os.sep

@st.cache_resource
def load_model():
    model     = joblib.load(MODEL_PATH + 'xgb_model.pkl')
    threshold = joblib.load(MODEL_PATH + 'threshold.pkl')
    with open(MODEL_PATH + 'train_stats.json', 'r') as f:
        train_stats = json.load(f)
    return model, threshold, train_stats

model, threshold, train_stats = load_model()

FEATURES = ['los', 'age', 'gender_male', 'high_risk_discharge',
            'prior_admissions', 'num_medications',
            'num_diagnoses', 'num_abnormal_labs']

# ── Validation ────────────────────────────────────────────
def validate(df):
    errors = []
    warns  = []
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    nulls = df[FEATURES].isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        warns.append(f"Missing values filled with median: {nulls.to_dict()}")
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df, errors, warns

# ── Drift detection ───────────────────────────────────────
def detect_drift(df):
    drifted = []
    for feature in FEATURES:
        train_mean  = train_stats[feature]['mean']
        upload_mean = df[feature].mean()
        diff_pct    = abs(upload_mean - train_mean) / (train_mean + 1e-9) * 100
        if diff_pct > 30:
            drifted.append({
                'feature'    : feature,
                'train_mean' : round(train_mean, 2),
                'upload_mean': round(upload_mean, 2),
                'diff_pct'   : round(diff_pct, 1)
            })
    return drifted

# ── Show patient ──────────────────────────────────────────
def show_patient(patient_X, score, tier):
    color = "🔴" if tier == 'HIGH' else "🟢"
    st.markdown(f"### {color} Risk Score: `{score:.2f}` — **{tier} RISK**")

    st.subheader("Why is this patient flagged?")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_X)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = patient_X.iloc[0].values,
            feature_names = FEATURES
        ), show=False
    )
    st.pyplot(fig)
    plt.close()

    st.subheader("What-if intervention simulator")
    st.markdown("Adjust sliders to simulate care changes.")
    new_los  = st.slider("Length of stay (days)", 0, 30,  int(patient_X['los'].values[0]))
    new_meds = st.slider("Number of medications", 0, 200, int(patient_X['num_medications'].values[0]))
    new_labs = st.slider("Abnormal lab count",    0, 500, int(patient_X['num_abnormal_labs'].values[0]))

    modified = patient_X.copy()
    modified['los']               = new_los
    modified['num_medications']   = new_meds
    modified['num_abnormal_labs'] = new_labs

    new_score = model.predict_proba(modified)[:, 1][0]
    delta     = new_score - score

    col1, col2 = st.columns(2)
    col1.metric("Original risk score",  f"{score:.2f}")
    col2.metric("Projected risk score", f"{new_score:.2f}", f"{delta:+.2f}")

    if delta < -0.05:
        pct = abs(delta / score) * 100
        st.success(f"✅ {pct:.0f}% risk reduction from interventions")
    elif delta > 0.05:
        st.error("⚠️ Risk increased — reconsider interventions")

# ── Session state ─────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# ── Auth pages ────────────────────────────────────────────
if not st.session_state.logged_in:
    st.title("🏥 CareSignal — Clinical Access Portal")
    st.markdown("*30-Day Readmission Risk Prediction for Verified Hospital Staff*")
    st.divider()

    tab_login, tab_register = st.tabs(["🔐 Login", "📝 Request Access"])

    with tab_login:
        st.subheader("Login to your account")
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if email and password:
                user = login_user(email, password)
                if user is None:
                    st.error("❌ Incorrect email or password")
                elif user['status'] == 'pending':
                    st.warning("⏳ Your account is pending approval. We will notify you within 24-48 hours.")
                elif user['status'] == 'approved':
                    st.session_state.logged_in = True
                    st.session_state.user      = user
                    st.rerun()
            else:
                st.warning("Please enter email and password")

    with tab_register:
        st.subheader("Request access to CareSignal")
        st.info("Access is restricted to verified hospital staff. Fill in the form below and we will review your application within 24-48 hours.")

        full_name  = st.text_input("Full name")
        email_reg  = st.text_input("Work email")
        password1  = st.text_input("Create password", type="password")
        password2  = st.text_input("Confirm password", type="password")
        hospital   = st.text_input("Hospital name")
        job_title  = st.text_input("Job title")
        reason     = st.text_area("Why do you need access?")

        if st.button("Submit application"):
            if not all([full_name, email_reg, password1, hospital, job_title, reason]):
                st.warning("Please fill in all fields")
            elif password1 != password2:
                st.error("Passwords do not match")
            else:
                success = register_user(
                    full_name, email_reg, password1,
                    hospital, job_title, reason
                )
                if success:
                    notify_admin(
                        full_name, email_reg,
                        hospital, job_title, reason
                    )
                    st.success("✅ Application submitted! We will review and approve within 24-48 hours.")
                else:
                    st.error("❌ Email already registered or error occurred")

    st.stop()

# ── Logged in ─────────────────────────────────────────────
user = st.session_state.user
st.sidebar.write(f"👤 {user['full_name']}")
st.sidebar.write(f"🏥 {user['hospital_name']}")
st.sidebar.write(f"💼 {user['job_title']}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user      = None
    st.rerun()

# ── Main app ──────────────────────────────────────────────
st.title("🏥 CareSignal — Readmission Risk")

tab1, tab2, tab3 = st.tabs([
    "📤 Upload & Score",
    "🔍 Patient Risk Explainer",
    "➕ New Patient Assessment"
])

# ── TAB 1: Upload & Score ─────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Upload discharge CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        df, errors, warns = validate(df)

        if errors:
            for e in errors: st.error(f"❌ {e}")
            st.stop()
        for w in warns: st.warning(f"⚠️ {w}")

        df['risk_score'] = model.predict_proba(df[FEATURES])[:, 1]
        df['risk_tier']  = df['risk_score'].apply(
            lambda x: 'HIGH' if x >= threshold else 'LOW')
        st.session_state['scored_df'] = df
        st.success(f"✅ {len(df)} patients scored")

        drifted = detect_drift(df)
        if drifted:
            st.warning("⚠️ Distribution drift detected — predictions may be less reliable.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total patients", len(df))
        col2.metric("High risk",      (df['risk_tier'] == 'HIGH').sum())
        col3.metric("Avg risk score", f"{df['risk_score'].mean():.2f}")

        st.subheader("📋 All patients — sorted by risk")
        st.dataframe(
            df[['hadm_id', 'risk_score', 'risk_tier']].sort_values(
                'risk_score', ascending=False),
            use_container_width=True
        )

        st.download_button(
            label     = "⬇ Download full risk report",
            data      = df[['hadm_id', 'risk_score', 'risk_tier']].sort_values(
                'risk_score', ascending=False).to_csv(index=False),
            file_name = "caresignal_risk_report.csv",
            mime      = "text/csv"
        )

    else:
        st.info("👆 Upload a discharge CSV to get started")
        st.markdown("""
        **Required columns:**
        `los` · `age` · `gender_male` · `high_risk_discharge` ·
        `prior_admissions` · `num_medications` · `num_diagnoses` · `num_abnormal_labs`
        """)
        template = pd.DataFrame(columns=FEATURES)
        st.download_button(
            label     = "📥 Download template CSV",
            data      = template.to_csv(index=False),
            file_name = "caresignal_template.csv",
            mime      = "text/csv"
        )

# ── TAB 2: Patient Risk Explainer ─────────────────────────
with tab2:
    if 'scored_df' not in st.session_state:
        st.info("👆 Upload a CSV in the Upload & Score tab first")
    else:
        df = st.session_state['scored_df']

        options    = ["— Select a patient —"] + df['hadm_id'].astype(str).tolist()
        patient_id = st.selectbox("Patient Admission ID", options)

        if patient_id == "— Select a patient —":
            st.info("Select a patient ID from the dropdown above.")
        else:
            patient   = df[df['hadm_id'].astype(str) == patient_id].iloc[0]
            patient_X = pd.DataFrame([patient[FEATURES]], columns=FEATURES)
            score     = patient['risk_score']
            tier      = patient['risk_tier']

            show_patient(patient_X, score, tier)

            st.divider()
            st.download_button(
                label     = f"⬇ Download report for patient {patient_id}",
                data      = pd.DataFrame([{
                    'hadm_id'          : patient_id,
                    'risk_score'       : score,
                    'risk_tier'        : tier,
                    'los'              : patient['los'],
                    'age'              : patient['age'],
                    'prior_admissions' : patient['prior_admissions'],
                    'num_medications'  : patient['num_medications'],
                    'num_diagnoses'    : patient['num_diagnoses'],
                    'num_abnormal_labs': patient['num_abnormal_labs'],
                }]).to_csv(index=False),
                file_name = f"patient_{patient_id}_report.csv",
                mime      = "text/csv"
            )

# ── TAB 3: New Patient Assessment ─────────────────────────
with tab3:
    st.subheader("Assess a new patient manually")
    st.markdown("Enter patient details below to get an instant readmission risk score.")

    col1, col2 = st.columns(2)
    with col1:
        age    = st.number_input("Age",                     0, 100, 0)
        los    = st.number_input("Length of stay (days)",   0, 60,  0)
        gender = st.selectbox("Gender", ["— Select —", "Male", "Female"])
        dc     = st.selectbox("Discharge to", ["— Select —", "Home", "SNF / Rehab"])
    with col2:
        prior  = st.number_input("Prior admissions (12mo)", 0, 20,  0)
        meds   = st.number_input("Number of medications",   0, 300, 0)
        diags  = st.number_input("Number of diagnoses",     0, 50,  0)
        labs   = st.number_input("Abnormal lab count",      0, 500, 0)

    if st.button("🔍 Get risk score"):
        if gender == "— Select —" or dc == "— Select —":
            st.warning("Please select gender and discharge destination")
        else:
            patient_X = pd.DataFrame([{
                'los'                : los,
                'age'                : age,
                'gender_male'        : 1 if gender == "Male" else 0,
                'high_risk_discharge': 1 if "SNF" in dc else 0,
                'prior_admissions'   : prior,
                'num_medications'    : meds,
                'num_diagnoses'      : diags,
                'num_abnormal_labs'  : labs
            }], columns=FEATURES)

            score = model.predict_proba(patient_X)[:, 1][0]
            tier  = 'HIGH' if score >= threshold else 'LOW'

            show_patient(patient_X, score, tier)

            st.divider()
            st.download_button(
                label     = "⬇ Download patient assessment",
                data      = pd.DataFrame([{
                    'risk_score'         : score,
                    'risk_tier'          : tier,
                    'age'                : age,
                    'los'                : los,
                    'prior_admissions'   : prior,
                    'num_medications'    : meds,
                    'num_diagnoses'      : diags,
                    'num_abnormal_labs'  : labs,
                    'gender_male'        : 1 if gender == "Male" else 0,
                    'high_risk_discharge': 1 if "SNF" in dc else 0,
                }]).to_csv(index=False),
                file_name = "patient_assessment.csv",
                mime      = "text/csv"
            )