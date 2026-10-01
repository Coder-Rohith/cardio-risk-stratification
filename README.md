# 🫀 CardioRisk AI — Preventive Health Risk Stratification System

**AI-Based Probabilistic Cardiovascular Risk Stratification Engine** using Isotonic-calibrated XGBoost, Random Survival Forests, and L2-regularized Logistic Regression with SHAP-based clinician explainability.

---

## 🏗️ Architecture

```
cardio-risk-stratification/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic cardiovascular data (Framingham-style)
│   ├── preprocessing.py      # Clinical feature engineering & scaling
│   ├── models.py             # XGBoost, RSF, Logistic Regression, Ensemble
│   ├── evaluation.py         # Stratified 5-Fold CV, ROC-AUC, Brier Score
│   ├── explainability.py     # SHAP TreeExplainer local/global attributions
│   └── train_pipeline.py     # End-to-end training orchestrator
├── app/
│   └── streamlit_app.py      # Real-time inference micro-application
├── data/                     # Generated datasets (auto-created)
└── models/                   # Serialized model artifacts (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd cardio-risk-stratification
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Train Models
```bash
python -m src.train_pipeline
```

### 3. Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## 📊 Models

| Model | Description |
|-------|-------------|
| **XGBoost (Isotonic)** | Gradient-boosted trees + isotonic regression calibration |
| **Random Survival Forest** | Time-to-event survival analysis via `scikit-survival` |
| **Logistic Regression (L2)** | Ridge-penalized linear baseline |
| **Ensemble** | Weighted average (0.50 XGB + 0.25 RSF + 0.25 LR) |

## 🎯 Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| ROC-AUC | ≥ 0.89 | ✓ (via Stratified 5-Fold CV) |
| Brier Score | ≤ 0.082 | ✓ (calibrated probabilities) |
| Inference Latency | < 200ms | ✓ (local CPU benchmark) |

## 🔍 Explainability

- **SHAP TreeExplainer** for XGBoost feature attributions
- Local explanations: waterfall plots, force plots per patient
- Global importance: mean |SHAP| ranking across all features
- Clinician-friendly risk factor / protective factor summaries

## 📋 Clinical Features (20 base + 10 engineered)

**Demographics**: age, sex  
**Vitals**: systolic_bp, diastolic_bp, heart_rate, bmi  
**Lipid Panel**: total_cholesterol, hdl_cholesterol, ldl_cholesterol, triglycerides  
**Metabolic**: fasting_glucose, hba1c, diabetes  
**Lifestyle**: smoking_status, physical_activity, alcohol_consumption  
**Medical History**: hypertension_med, family_history_cvd  
**Labs**: crp_level, creatinine  
**Engineered**: chol_hdl_ratio, ldl_hdl_ratio, tg_hdl_ratio, pulse_pressure, mean_arterial_pressure, metabolic_risk, age_sbp_product, age_chol_product, bmi_category, bp_stage
