"""
Streamlit Micro-Application for Cardiovascular Risk Stratification
===================================================================
Real-time interpretable inference with SHAP explanations.
Target: <200ms latency per prediction on local CPU.

Features:
- Patient input form with clinical parameters
- Real-time risk prediction with gauge visualization
- SHAP-based feature attribution (waterfall + force plots)
- Global feature importance dashboard
- Model comparison view
- Risk stratification categories (Low/Moderate/High/Very High)
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ── Project Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"

from src.preprocessing import (
    ClinicalDataPreprocessor,
    engineer_features,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.explainability import SHAPExplainer


# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CardioRisk AI · Risk Stratification Engine",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════
# CUSTOM THEME CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #0a0e1a;
        --bg-card: #111827;
        --bg-card-hover: #1a2332;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-color: #1e293b;
        --gradient-start: #3b82f6;
        --gradient-end: #06b6d4;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
    }
    .main-header h1 {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1em;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #111827, #1a2332);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        margin: 4px 0;
        line-height: 1.2;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85em;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        letter-spacing: 0.03em;
    }
    .risk-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .risk-moderate { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .risk-high { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .risk-very-high { background: rgba(220, 38, 38, 0.2); color: #fca5a5; border: 1px solid rgba(220, 38, 38, 0.4); }

    /* Section Header */
    .section-header {
        color: #f1f5f9;
        font-size: 1.3em;
        font-weight: 700;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    }

    /* Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }

    /* Feature explanation */
    .feature-row {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(30, 41, 59, 0.5);
    }
    .feature-name { flex: 1; color: #e2e8f0; font-weight: 500; }
    .feature-impact-pos { color: #f87171; font-weight: 600; }
    .feature-impact-neg { color: #34d399; font-weight: 600; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* Hide Streamlit defaults */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
    }

    .latency-badge {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CACHING: Load Models
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Load all trained model artifacts."""
    models_dir = MODELS_DIR

    artifacts = {}
    try:
        artifacts["preprocessor"] = ClinicalDataPreprocessor().load(
            str(models_dir / "preprocessor.joblib")
        )
        artifacts["xgboost"] = joblib.load(str(models_dir / "xgboost_isotonic.joblib"))
        artifacts["rsf"] = joblib.load(str(models_dir / "rsf_model.joblib"))
        artifacts["logistic"] = joblib.load(str(models_dir / "logistic_l2.joblib"))
        artifacts["ensemble"] = joblib.load(str(models_dir / "ensemble_model.joblib"))
        artifacts["background_data"] = np.load(str(models_dir / "background_data.npy"))

        metadata_path = models_dir / "pipeline_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                artifacts["metadata"] = json.load(f)
        else:
            artifacts["metadata"] = {}

        importance_path = models_dir / "feature_importance.csv"
        if importance_path.exists():
            artifacts["feature_importance"] = pd.read_csv(importance_path)

        artifacts["loaded"] = True
    except FileNotFoundError as e:
        st.error(f"Model files not found. Run `train_pipeline.py` first.\n\n{e}")
        artifacts["loaded"] = False

    return artifacts


@st.cache_resource
def load_shap_explainer(_artifacts):
    """Load/create SHAP explainer (cached separately for performance)."""
    if not _artifacts.get("loaded"):
        return None
    try:
        feature_names = _artifacts["metadata"].get(
            "feature_names", _artifacts["preprocessor"].feature_names
        )
        explainer = SHAPExplainer(
            _artifacts["xgboost"],
            feature_names,
            model_type="xgboost",
            background_data=_artifacts["background_data"],
        )
        return explainer
    except Exception as e:
        st.warning(f"SHAP explainer initialization warning: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def classify_risk(probability: float) -> dict:
    """Classify risk into clinical categories."""
    if probability < 0.10:
        return {"level": "Low", "class": "risk-low", "color": "#10b981", "advice": "Maintain healthy lifestyle. Routine screening."}
    elif probability < 0.20:
        return {"level": "Moderate", "class": "risk-moderate", "color": "#f59e0b", "advice": "Lifestyle modifications recommended. Consider statin therapy evaluation."}
    elif probability < 0.35:
        return {"level": "High", "class": "risk-high", "color": "#ef4444", "advice": "Aggressive risk factor management. Statin + antihypertensive therapy indicated."}
    else:
        return {"level": "Very High", "class": "risk-very-high", "color": "#dc2626", "advice": "Immediate clinical intervention. Comprehensive cardiology referral."}


def create_risk_gauge(probability: float, risk_info: dict) -> go.Figure:
    """Create a premium risk gauge visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "#f1f5f9", "family": "Inter"}},
        title={"text": "Cardiovascular Risk Score", "font": {"size": 16, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickwidth": 1},
            "bar": {"color": risk_info["color"], "thickness": 0.3},
            "bgcolor": "rgba(17, 24, 39, 0.8)",
            "borderwidth": 2,
            "bordercolor": "#1e293b",
            "steps": [
                {"range": [0, 10], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [10, 20], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [20, 35], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [35, 100], "color": "rgba(220, 38, 38, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 3},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def create_model_comparison_chart(predictions: dict) -> go.Figure:
    """Create model comparison bar chart."""
    models = list(predictions.keys())
    probs = [v * 100 for v in predictions.values()]
    colors = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981"]

    fig = go.Figure()
    for i, (model, prob) in enumerate(zip(models, probs)):
        fig.add_trace(go.Bar(
            x=[model],
            y=[prob],
            marker_color=colors[i % len(colors)],
            marker_line=dict(color=colors[i % len(colors)], width=1),
            text=[f"{prob:.1f}%"],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=13, family="Inter"),
            name=model,
            showlegend=False,
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            title="Risk Probability (%)",
            range=[0, max(probs) * 1.3 + 5],
            gridcolor="rgba(30, 41, 59, 0.5)",
            color="#94a3b8",
        ),
        xaxis=dict(color="#94a3b8"),
        font=dict(family="Inter"),
    )
    return fig


def create_shap_bar_chart(explanations: list, top_n: int = 10) -> go.Figure:
    """Create a horizontal bar chart of SHAP feature contributions."""
    top = explanations[:top_n]
    top.reverse()

    features = [e["feature"] for e in top]
    values = [e["shap_value"] for e in top]
    colors = ["#f87171" if v > 0 else "#34d399" for v in values]

    fig = go.Figure(go.Bar(
        y=features,
        x=values,
        orientation="h",
        marker_color=colors,
        marker_line=dict(color=colors, width=1),
        text=[f"{abs(v):.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11, family="Inter"),
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=60, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="SHAP Value (impact on risk)",
            gridcolor="rgba(30, 41, 59, 0.5)",
            color="#94a3b8",
            zeroline=True,
            zerolinecolor="#475569",
        ),
        yaxis=dict(color="#e2e8f0"),
        font=dict(family="Inter"),
    )
    return fig


def create_global_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create global feature importance visualization."""
    top = importance_df.head(top_n).iloc[::-1]

    fig = go.Figure(go.Bar(
        y=top["feature"],
        x=top["mean_abs_shap"],
        orientation="h",
        marker=dict(
            color=top["mean_abs_shap"],
            colorscale=[[0, "#06b6d4"], [0.5, "#3b82f6"], [1, "#8b5cf6"]],
            line=dict(width=0),
        ),
        text=[f"{v:.4f}" for v in top["mean_abs_shap"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11, family="Inter"),
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=80, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Mean |SHAP Value|",
            gridcolor="rgba(30, 41, 59, 0.5)",
            color="#94a3b8",
        ),
        yaxis=dict(color="#e2e8f0"),
        font=dict(family="Inter"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫀 CardioRisk AI</h1>
        <p>Probabilistic Cardiovascular Risk Stratification Engine · XGBoost + RSF + Logistic Regression · SHAP Explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    artifacts = load_models()
    if not artifacts.get("loaded"):
        st.error("⚠️ Models not trained yet. Run the training pipeline first:")
        st.code("cd E:/General/cardio-risk-stratification && python -m src.train_pipeline", language="bash")
        return

    shap_explainer = load_shap_explainer(artifacts)
    metadata = artifacts.get("metadata", {})
    feature_names = metadata.get("feature_names", artifacts["preprocessor"].feature_names)

    # ── Sidebar: Patient Input Form ──────────────────────────────────
    with st.sidebar:
        st.markdown("### 📋 Patient Clinical Data")
        st.markdown("---")

        st.markdown("**Demographics**")
        age = st.slider("Age (years)", 30, 85, 55, key="age")
        sex = st.selectbox("Sex", ["Female", "Male"], index=1, key="sex")
        sex_val = 1 if sex == "Male" else 0

        st.markdown("---")
        st.markdown("**Vitals**")
        systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 130, key="sbp")
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 55, 120, 80, key="dbp")
        heart_rate = st.slider("Heart Rate (bpm)", 50, 110, 72, key="hr")
        bmi = st.slider("BMI (kg/m²)", 16.0, 45.0, 27.5, step=0.5, key="bmi")

        st.markdown("---")
        st.markdown("**Lipid Panel**")
        total_cholesterol = st.slider("Total Cholesterol (mg/dL)", 120, 350, 210, key="tc")
        hdl_cholesterol = st.slider("HDL Cholesterol (mg/dL)", 25, 100, 50, key="hdl")
        ldl_cholesterol = st.slider("LDL Cholesterol (mg/dL)", 40, 250, 130, key="ldl")
        triglycerides = st.slider("Triglycerides (mg/dL)", 50, 500, 150, key="tg")

        st.markdown("---")
        st.markdown("**Metabolic**")
        fasting_glucose = st.slider("Fasting Glucose (mg/dL)", 65, 300, 95, key="fg")
        hba1c = st.slider("HbA1c (%)", 4.0, 14.0, 5.5, step=0.1, key="hba1c")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], key="dm")
        diabetes_val = 1 if diabetes == "Yes" else 0

        st.markdown("---")
        st.markdown("**Lifestyle**")
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], key="smoke")
        smoking_val = {"Never": 0, "Former": 1, "Current": 2}[smoking]
        activity = st.selectbox("Physical Activity", ["Sedentary", "Light", "Moderate", "Vigorous"], index=2, key="pa")
        activity_val = {"Sedentary": 0, "Light": 1, "Moderate": 2, "Vigorous": 3}[activity]
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"], index=1, key="alc")
        alcohol_val = {"None": 0, "Moderate": 1, "Heavy": 2}[alcohol]

        st.markdown("---")
        st.markdown("**Medical History & Labs**")
        hypertension_med = st.selectbox("On Hypertension Medication", ["No", "Yes"], key="htn")
        htn_val = 1 if hypertension_med == "Yes" else 0
        family_history = st.selectbox("Family History of CVD", ["No", "Yes"], key="fhx")
        fhx_val = 1 if family_history == "Yes" else 0
        crp_level = st.slider("CRP Level (mg/L)", 0.1, 15.0, 1.5, step=0.1, key="crp")
        creatinine = st.slider("Creatinine (mg/dL)", 0.5, 3.0, 1.0, step=0.05, key="cr")

        st.markdown("---")
        predict_btn = st.button("🔬 Analyze Risk", use_container_width=True, type="primary")

    # ── Build Patient DataFrame ──────────────────────────────────────
    patient_data = pd.DataFrame([{
        "age": age,
        "sex": sex_val,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "bmi": bmi,
        "total_cholesterol": total_cholesterol,
        "hdl_cholesterol": hdl_cholesterol,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "fasting_glucose": fasting_glucose,
        "hba1c": hba1c,
        "smoking_status": smoking_val,
        "physical_activity": activity_val,
        "alcohol_consumption": alcohol_val,
        "diabetes": diabetes_val,
        "hypertension_med": htn_val,
        "family_history_cvd": fhx_val,
        "crp_level": crp_level,
        "creatinine": creatinine,
    }])

    # ── Predict ──────────────────────────────────────────────────────
    # Always predict (sidebar changes trigger re-run)
    t0 = time.perf_counter()
    X_patient = artifacts["preprocessor"].transform(patient_data)
    ensemble_prob = artifacts["ensemble"].predict_proba(X_patient)[0, 1]
    latency_ms = (time.perf_counter() - t0) * 1000

    # Individual model predictions
    xgb_prob = artifacts["xgboost"].predict_proba(X_patient)[0, 1]
    rsf_prob = artifacts["rsf"].predict_proba(X_patient)[0, 1]
    lr_prob = artifacts["logistic"].predict_proba(X_patient)[0, 1]

    risk_info = classify_risk(ensemble_prob)

    # ── MAIN CONTENT AREA ────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🎯 Risk Assessment",
        "🔍 SHAP Explanations",
        "📊 Global Analytics",
    ])

    with tab1:
        # ── Top Metrics Row ──────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Ensemble Risk</div>
                <div class="metric-value" style="color: {risk_info['color']}">{ensemble_prob*100:.1f}%</div>
                <span class="risk-badge {risk_info['class']}">{risk_info['level']}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">XGBoost Score</div>
                <div class="metric-value" style="color: #3b82f6">{xgb_prob*100:.1f}%</div>
                <div class="metric-label">Isotonic Calibrated</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RSF Score</div>
                <div class="metric-value" style="color: #06b6d4">{rsf_prob*100:.1f}%</div>
                <div class="metric-label">Survival Forest</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Inference Latency</div>
                <div class="metric-value" style="color: #10b981">{latency_ms:.0f}ms</div>
                <span class="latency-badge">{"✓ Under 200ms" if latency_ms < 200 else "⚠ Above target"}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk Gauge + Model Comparison ────────────────────────────
        gc1, gc2 = st.columns([1, 1])

        with gc1:
            st.markdown('<div class="section-header">Risk Score Gauge</div>', unsafe_allow_html=True)
            gauge_fig = create_risk_gauge(ensemble_prob, risk_info)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with gc2:
            st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
            comp_fig = create_model_comparison_chart({
                "XGBoost\n(Isotonic)": xgb_prob,
                "Random\nSurvival Forest": rsf_prob,
                "Logistic\nRegression (L2)": lr_prob,
                "Ensemble\n(Weighted)": ensemble_prob,
            })
            st.plotly_chart(comp_fig, use_container_width=True)

        # ── Clinical Recommendation ──────────────────────────────────
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 1.5em;">🏥</span>
                <span style="color: #f1f5f9; font-weight: 700; font-size: 1.1em;">Clinical Recommendation</span>
                <span class="risk-badge {risk_info['class']}">{risk_info['level']} Risk</span>
            </div>
            <p style="color: #cbd5e1; margin: 0; line-height: 1.6;">{risk_info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">SHAP Feature Attributions (Local Explanation)</div>', unsafe_allow_html=True)

        if shap_explainer is not None:
            try:
                patient_explanation = shap_explainer.explain_patient(
                    X_patient[0], X_background=artifacts["background_data"]
                )

                # SHAP bar chart
                shap_fig = create_shap_bar_chart(patient_explanation["explanations"], top_n=12)
                st.plotly_chart(shap_fig, use_container_width=True)

                # Top risk / protective factors
                rc1, rc2 = st.columns(2)

                with rc1:
                    st.markdown("""
                    <div class="glass-card">
                        <div style="color: #f87171; font-weight: 700; font-size: 1.1em; margin-bottom: 12px;">
                            ⚠️ Top Risk Factors
                        </div>
                    """, unsafe_allow_html=True)

                    risk_factors = [e for e in patient_explanation["explanations"] if e["shap_value"] > 0][:5]
                    for rf in risk_factors:
                        st.markdown(f"""
                        <div class="feature-row">
                            <span class="feature-name">{rf['feature']}</span>
                            <span class="feature-impact-pos">+{rf['shap_value']:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                with rc2:
                    st.markdown("""
                    <div class="glass-card">
                        <div style="color: #34d399; font-weight: 700; font-size: 1.1em; margin-bottom: 12px;">
                            🛡️ Top Protective Factors
                        </div>
                    """, unsafe_allow_html=True)

                    protective = [e for e in patient_explanation["explanations"] if e["shap_value"] < 0][:5]
                    for pf in protective:
                        st.markdown(f"""
                        <div class="feature-row">
                            <span class="feature-name">{pf['feature']}</span>
                            <span class="feature-impact-neg">{pf['shap_value']:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # Waterfall plot
                st.markdown('<div class="section-header">SHAP Waterfall Plot</div>', unsafe_allow_html=True)
                try:
                    shap_vals_patient = shap_explainer.explainer.shap_values(X_patient)
                    if isinstance(shap_vals_patient, list):
                        shap_vals_patient = shap_vals_patient[1]
                    shap_vals_patient = shap_vals_patient.flatten()

                    ev = shap_explainer.expected_value if shap_explainer.expected_value is not None else 0.0

                    explanation_obj = shap.Explanation(
                        values=shap_vals_patient,
                        base_values=ev,
                        data=X_patient[0],
                        feature_names=feature_names,
                    )

                    fig_wf, ax_wf = plt.subplots(figsize=(12, 6))
                    shap.plots.waterfall(explanation_obj, show=False)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception as e:
                    st.info(f"Waterfall plot note: {e}")

            except Exception as e:
                st.warning(f"SHAP explanation note: {e}")
        else:
            st.info("SHAP explanations are available after model training.")

    with tab3:
        st.markdown('<div class="section-header">Global Feature Importance (Mean |SHAP|)</div>', unsafe_allow_html=True)

        if "feature_importance" in artifacts:
            imp_fig = create_global_importance_chart(artifacts["feature_importance"])
            st.plotly_chart(imp_fig, use_container_width=True)

            # Show table
            st.markdown('<div class="section-header">Feature Importance Table</div>', unsafe_allow_html=True)
            st.dataframe(
                artifacts["feature_importance"][["rank", "feature", "mean_abs_shap", "pct_contribution"]].head(20),
                use_container_width=True,
                hide_index=True,
            )

        # Pipeline metadata
        st.markdown('<div class="section-header">Pipeline Metadata</div>', unsafe_allow_html=True)
        if metadata:
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Training Samples", f"{metadata.get('train_size', 'N/A'):,}")
            with mc2:
                st.metric("Test Samples", f"{metadata.get('test_size', 'N/A'):,}")
            with mc3:
                st.metric("Features", metadata.get("n_features", "N/A"))
            with mc4:
                st.metric("CV Folds", metadata.get("cv_folds", "N/A"))

            # CV Results
            cv_summary = metadata.get("cv_summary", {})
            if cv_summary:
                st.markdown('<div class="section-header">Cross-Validation Results</div>', unsafe_allow_html=True)
                cv_rows = []
                for model_key, metrics in cv_summary.items():
                    display_name = {
                        "xgboost": "XGBoost (Isotonic)",
                        "rsf": "Random Survival Forest",
                        "logistic": "Logistic Regression (L2)",
                        "ensemble": "Ensemble (Weighted)",
                    }.get(model_key, model_key)
                    cv_rows.append({
                        "Model": display_name,
                        "ROC-AUC (mean)": metrics.get("roc_auc_mean", "N/A"),
                        "ROC-AUC (std)": metrics.get("roc_auc_std", "N/A"),
                        "Brier Score (mean)": metrics.get("brier_mean", "N/A"),
                        "Brier Score (std)": metrics.get("brier_std", "N/A"),
                    })
                st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

            # Holdout results
            holdout = metadata.get("holdout_metrics", {})
            if holdout:
                st.markdown('<div class="section-header">Holdout Test Results</div>', unsafe_allow_html=True)
                holdout_rows = []
                for model_name, metrics in holdout.items():
                    holdout_rows.append({
                        "Model": model_name,
                        "ROC-AUC": metrics.get("roc_auc", "N/A"),
                        "Brier Score": metrics.get("brier_score", "N/A"),
                    })
                st.dataframe(pd.DataFrame(holdout_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
