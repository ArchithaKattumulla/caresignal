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

```
caresignal/
├── app.py              
├── outputs/            
├── data/               
└── notebooks/          
```

## Model Performance
Trained on MIMIC-IV demo dataset (100 patients).
- AUC-ROC: 0.911
- AUC-PR: 0.892
- Brier Score: 0.895

Architecture designed to scale to full 500k admission dataset 
upon PhysioNet credentialing approval.

## Author
Architha Kattumulla — Montclair State University, 2026
