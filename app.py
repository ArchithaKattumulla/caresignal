import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# ── Simple Login ──────────────────────────────────────────
USERS = {
    "nurse1"  : "nurse123",
    "doctor1" : "doctor123",
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🏥 CareSignal — Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username  = username
            st.rerun()
        else:
            st.error("❌ Incorrect username or password")
    st.stop()

# ── Logged in ─────────────────────────────────────────────
st.sidebar.write(f"👤 {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ── Load model ────────────────────────────────────────────
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'outputs') + os.sep
model      = joblib.load(MODEL_PATH + 'xgb_model.pkl')
threshold  = joblib.load(MODEL_PATH + 'threshold.pkl')

with open(MODEL_PATH + 'train_stats.json', 'r') as f:
    train_stats = json.load(f)

FEATURES = ['los', 'age', 'gender_male', 'high_risk_discharge',
            'prior_admissions', 'num_medications',
            'num_diagnoses', 'num_abnormal_labs']

# ── Validation ────────────────────────────────────────────
def validate(df):
    errors  = []
    warns   = []
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

# ── Show patient results ──────────────────────────────────
def show_patient(patient_X, score, tier):
    st.metric("Risk Score", f"{score:.2f}", tier)

    # SHAP
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

    # what-if
    st.subheader("What-if simulator")
    new_los  = st.slider("Length of stay",  0, 30,  int(patient_X['los'].values[0]))
    new_meds = st.slider("Medications",     0, 200, int(patient_X['num_medications'].values[0]))
    new_labs = st.slider("Abnormal labs",   0, 500, int(patient_X['num_abnormal_labs'].values[0]))

    modified = patient_X.copy()
    modified['los']               = new_los
    modified['num_medications']   = new_meds
    modified['num_abnormal_labs'] = new_labs

    new_score = model.predict_proba(modified)[:, 1][0]
    delta     = new_score - score
    st.metric("New risk score", f"{new_score:.2f}", f"{delta:+.2f}")

# ── Title ─────────────────────────────────────────────────
st.title("🏥 CareSignal — Readmission Risk")

tab1, tab2 = st.tabs(["📤 Upload CSV", "👤 Single Patient"])

# ── TAB 1: Upload CSV ─────────────────────────────────────
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
        st.success(f"✅ {len(df)} patients scored")

        drifted = detect_drift(df)
        if drifted:
            st.warning("⚠️ Drift detected — predictions may be less reliable")
            st.dataframe(pd.DataFrame(drifted))

        col1, col2, col3 = st.columns(3)
        col1.metric("Total patients", len(df))
        col2.metric("High risk",      (df['risk_tier'] == 'HIGH').sum())
        col3.metric("Avg risk score", f"{df['risk_score'].mean():.2f}")


        # download report
        st.subheader("📥 Download risk report")
        report = df[['hadm_id', 'risk_score', 'risk_tier']].sort_values(
        'risk_score', ascending=False)
        st.download_button(
          label     = "⬇ Download full risk report",
          data      = report.to_csv(index=False),
          file_name = "caresignal_risk_report.csv",
          mime      = "text/csv"
        )



        st.subheader("📋 All patients — sorted by risk")
        st.dataframe(
            df[['hadm_id', 'risk_score', 'risk_tier']].sort_values(
                'risk_score', ascending=False),
            use_container_width=True
        )
        st.divider()

        patient_id = st.selectbox(
            "Select patient by Admission ID",
            df['hadm_id'].astype(str).tolist()
        )
        patient   = df[df['hadm_id'].astype(str) == patient_id].iloc[0]
        patient_X = pd.DataFrame([patient[FEATURES]], columns=FEATURES)
        score     = patient['risk_score']
        tier      = patient['risk_tier']
        show_patient(patient_X, score, tier)

        # download individual patient report
        st.subheader("📥 Download patient report")
        patient_report = pd.DataFrame([{
    'hadm_id'   : patient_id,
    'risk_score': score,
    'risk_tier' : tier,
    'los'       : patient['los'],
    'age'       : patient['age'],
    'prior_admissions'  : patient['prior_admissions'],
    'num_medications'   : patient['num_medications'],
    'num_diagnoses'     : patient['num_diagnoses'],
    'num_abnormal_labs' : patient['num_abnormal_labs'],
}])
        st.download_button(
          label     = f"⬇ Download report for patient {patient_id}",
          data      = patient_report.to_csv(index=False),
          file_name = f"patient_{patient_id}_report.csv",
          mime      = "text/csv"
        )

    else:
        st.info("👆 Upload a discharge CSV to get started")
        template = pd.DataFrame(columns=FEATURES)
        st.download_button(
            label     = "📥 Download template CSV",
            data      = template.to_csv(index=False),
            file_name = "caresignal_template.csv",
            mime      = "text/csv"
        )

# ── TAB 2: Single Patient ─────────────────────────────────
with tab2:
    st.subheader("Enter patient details manually")

    col1, col2 = st.columns(2)
    with col1:
        age    = st.number_input("Age",                    10, 100, 65)
        los    = st.number_input("Length of stay (days)",  0,  60,  5)
        gender = st.selectbox("Gender", ["Male", "Female"])
        dc     = st.selectbox("Discharge to", ["Home", "SNF / Rehab"])
    with col2:
        prior  = st.number_input("Prior admissions (12mo)", 0, 20,  0)
        meds   = st.number_input("Number of medications",   0, 300, 10)
        diags  = st.number_input("Number of diagnoses",     0, 50,  5)
        labs   = st.number_input("Abnormal lab count",      0, 500, 20)

    if st.button("🔍 Get risk score"):
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