# 🏥 CareSignal — 30-Day Hospital Readmission Risk Prediction

A clinical machine learning system that predicts 30-day hospital readmission 
risk at the patient level. Hospital discharge staff upload their weekly patient 
report and receive risk scores, SHAP-based explanations, and intervention 
simulations in real time.

## Problem
Hospitals face financial penalties under the Hospital Readmissions Reduction 
Program (HRRP) for excessive readmission rates. This tool helps discharge 
teams identify high-risk patients before they leave the hospital.

## Features
- Upload weekly discharge CSV → score all patients instantly
- Per-patient SHAP explanation — exactly why each patient was flagged
- What-if intervention simulator — see how care changes affect risk score
- Distribution drift detection — warns when uploaded data differs from training
- Role-based login for hospital staff
- Download risk report as CSV

## Tech Stack
- **Data** — MIMIC-IV Clinical Database (PhysioNet)
- **Model** — XGBoost tuned with Optuna
- **Explainability** — SHAP waterfall plots
- **Dashboard** — Streamlit
- **Deployment** — ngrok

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
