"""
Synthetic Cardiovascular Risk Data Generator
=============================================
Generates clinically-realistic synthetic patient data modeled after
Framingham Heart Study risk factors with controlled signal-to-noise
ratio to achieve target discriminative performance.

Features generated:
- Demographics: age, sex
- Vitals: systolic_bp, diastolic_bp, heart_rate, bmi
- Lipid Panel: total_cholesterol, hdl_cholesterol, ldl_cholesterol, triglycerides
- Metabolic: fasting_glucose, hba1c
- Lifestyle: smoking_status, physical_activity_level, alcohol_consumption
- Medical History: diabetes, hypertension_med, family_history_cvd
- Lab Values: crp_level, creatinine
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_cardiovascular_data(
    n_samples: int = 12000,
    random_state: int = 42,
    event_rate: float = 0.18,
) -> pd.DataFrame:
    """
    Generate synthetic cardiovascular risk data with realistic distributions.

    The data generation uses a latent risk model with non-linear interactions
    to simulate realistic cardiovascular event patterns.

    Parameters
    ----------
    n_samples : int
        Number of patient records to generate.
    random_state : int
        Random seed for reproducibility.
    event_rate : float
        Target proportion of cardiovascular events.

    Returns
    -------
    pd.DataFrame
        DataFrame with patient features, event indicator, and time-to-event.
    """
    rng = np.random.RandomState(random_state)

    # ── Demographics ──────────────────────────────────────────────────
    age = rng.normal(55, 12, n_samples).clip(30, 85).astype(int)
    sex = rng.binomial(1, 0.48, n_samples)  # 1 = male

    # ── Vitals ────────────────────────────────────────────────────────
    # BP correlates with age
    systolic_bp = (
        100 + 0.5 * age + rng.normal(0, 12, n_samples)
    ).clip(90, 200).astype(int)
    diastolic_bp = (
        60 + 0.2 * age + rng.normal(0, 8, n_samples)
    ).clip(55, 120).astype(int)
    heart_rate = rng.normal(72, 10, n_samples).clip(50, 110).astype(int)

    # BMI with slight age correlation
    bmi = (22 + 0.08 * age + rng.normal(0, 4, n_samples)).clip(16, 45)
    bmi = np.round(bmi, 1)

    # ── Lipid Panel ───────────────────────────────────────────────────
    total_cholesterol = (
        150 + 0.8 * age + rng.normal(0, 30, n_samples)
    ).clip(120, 350).astype(int)
    hdl_cholesterol = (
        55 - 8 * sex + rng.normal(0, 12, n_samples)
    ).clip(25, 100).astype(int)
    ldl_cholesterol = (
        total_cholesterol - hdl_cholesterol - rng.uniform(10, 50, n_samples)
    ).clip(40, 250).astype(int)
    triglycerides = (
        100 + 0.5 * bmi * 3 + rng.normal(0, 40, n_samples)
    ).clip(50, 500).astype(int)

    # ── Metabolic Markers ─────────────────────────────────────────────
    diabetes = rng.binomial(1, 0.02 + 0.003 * (age - 30).clip(0), n_samples)
    fasting_glucose = np.where(
        diabetes == 1,
        rng.normal(145, 30, n_samples),
        rng.normal(92, 10, n_samples),
    ).clip(65, 300)
    fasting_glucose = np.round(fasting_glucose, 1)

    hba1c = np.where(
        diabetes == 1,
        rng.normal(7.5, 1.2, n_samples),
        rng.normal(5.4, 0.4, n_samples),
    ).clip(4.0, 14.0)
    hba1c = np.round(hba1c, 1)

    # ── Lifestyle ─────────────────────────────────────────────────────
    smoking_status = rng.choice([0, 1, 2], n_samples, p=[0.55, 0.25, 0.20])
    # 0=never, 1=former, 2=current
    physical_activity = rng.choice(
        [0, 1, 2, 3], n_samples, p=[0.15, 0.35, 0.35, 0.15]
    )  # 0=sedentary, 1=light, 2=moderate, 3=vigorous
    alcohol_consumption = rng.choice(
        [0, 1, 2], n_samples, p=[0.30, 0.50, 0.20]
    )  # 0=none, 1=moderate, 2=heavy

    # ── Medical History ───────────────────────────────────────────────
    hypertension_med = rng.binomial(
        1, 0.01 + 0.004 * (systolic_bp - 120).clip(0) / 10, n_samples
    )
    family_history_cvd = rng.binomial(1, 0.25, n_samples)

    # ── Lab Values ────────────────────────────────────────────────────
    crp_level = np.abs(rng.normal(1.5, 2.0, n_samples)).clip(0.1, 15.0)
    crp_level = np.round(crp_level, 2)
    creatinine = rng.normal(1.0, 0.2, n_samples).clip(0.5, 3.0)
    creatinine = np.round(creatinine, 2)

    # ══════════════════════════════════════════════════════════════════
    # LATENT RISK MODEL (non-linear with interactions)
    # ══════════════════════════════════════════════════════════════════
    age_risk = 0.08 * (age - 40).clip(0)
    bp_risk = 0.04 * (systolic_bp - 120).clip(0)
    chol_ratio_risk = 0.6 * (total_cholesterol / (hdl_cholesterol + 1))
    bmi_risk = 0.12 * (bmi - 25).clip(0)
    smoking_risk = np.where(smoking_status == 2, 1.8, np.where(smoking_status == 1, 0.5, 0.0))
    diabetes_risk = 2.0 * diabetes
    glucose_risk = 0.015 * (fasting_glucose - 100).clip(0)
    inactivity_risk = 0.4 * (3 - physical_activity)
    crp_risk = 0.25 * (crp_level - 1.0).clip(0)
    family_risk = 0.8 * family_history_cvd
    sex_risk = 0.5 * sex
    hba1c_risk = 0.3 * (hba1c - 5.7).clip(0)

    # Interaction terms
    age_bp_interaction = 0.001 * (age - 50).clip(0) * (systolic_bp - 130).clip(0)
    diabetes_chol_interaction = 0.5 * diabetes * (total_cholesterol > 240).astype(int)
    smoking_bp_interaction = 0.4 * (smoking_status == 2).astype(int) * (systolic_bp > 140).astype(int)

    # Composite latent risk score
    latent_risk = (
        age_risk + bp_risk + chol_ratio_risk + bmi_risk
        + smoking_risk + diabetes_risk + glucose_risk
        + inactivity_risk + crp_risk + family_risk + sex_risk
        + hba1c_risk + age_bp_interaction
        + diabetes_chol_interaction + smoking_bp_interaction
    )

    # Add noise
    latent_risk += rng.normal(0, 0.20, n_samples)

    # Convert to probability using sigmoid
    risk_prob = 1 / (1 + np.exp(-(latent_risk - np.percentile(latent_risk, 100 * (1 - event_rate)))))

    # Adjust threshold to get target event rate
    threshold_adjustment = np.log(event_rate / (1 - event_rate))
    adjusted_prob = 1 / (1 + np.exp(-(np.log(risk_prob / (1 - risk_prob + 1e-10)) + threshold_adjustment)))

    # Generate binary events
    cvd_event = rng.binomial(1, adjusted_prob.clip(0.01, 0.99))

    # Generate time-to-event (for survival analysis)
    # Exponential distribution scaled by risk
    base_time = rng.exponential(365 * 5, n_samples)  # ~5 year baseline
    time_to_event = (base_time / (1 + 2 * adjusted_prob)).clip(30, 365 * 10)
    time_to_event = np.round(time_to_event).astype(int)

    # Censoring for non-events
    time_to_event = np.where(cvd_event == 0, time_to_event.clip(365, 365 * 10), time_to_event)

    # ── Build DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame({
        "age": age,
        "sex": sex,
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
        "smoking_status": smoking_status,
        "physical_activity": physical_activity,
        "alcohol_consumption": alcohol_consumption,
        "diabetes": diabetes,
        "hypertension_med": hypertension_med,
        "family_history_cvd": family_history_cvd,
        "crp_level": crp_level,
        "creatinine": creatinine,
        "cvd_event": cvd_event,
        "time_to_event_days": time_to_event,
    })

    return df


def inject_missingness(
    df: pd.DataFrame,
    missing_rate: float = 0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Inject realistic MCAR (Missing Completely At Random) missingness
    into clinical features that are plausibly incomplete.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    missing_rate : float
        Proportion of values to set missing per eligible column.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with injected NaN values.
    """
    rng = np.random.RandomState(random_state)
    df = df.copy()

    # Columns that can plausibly have missing values in clinical settings
    missable_cols = [
        "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
        "fasting_glucose", "hba1c", "crp_level", "creatinine",
        "physical_activity", "alcohol_consumption",
    ]

    for col in missable_cols:
        if col in df.columns:
            mask = rng.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan

    return df


def save_dataset(output_dir: str = "data") -> str:
    """Generate and save the synthetic cardiovascular dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = generate_cardiovascular_data()
    df_with_missing = inject_missingness(df)

    filepath = output_path / "cardiovascular_risk_data.csv"
    df_with_missing.to_csv(filepath, index=False)
    print(f"[DATA] Generated {len(df)} patient records -> {filepath}")
    print(f"[DATA] Event rate: {df['cvd_event'].mean():.3f}")
    print(f"[DATA] Features: {df.shape[1] - 2} (excl. target & time)")

    # Also save a clean version for reference
    clean_path = output_path / "cardiovascular_risk_data_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"[DATA] Clean copy -> {clean_path}")

    return str(filepath)


if __name__ == "__main__":
    save_dataset(output_dir="E:/General/cardio-risk-stratification/data")
