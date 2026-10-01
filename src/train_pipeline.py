"""
Main Training Pipeline
========================
End-to-end pipeline that:
1. Generates/loads synthetic cardiovascular data
2. Preprocesses and engineers features
3. Runs Stratified 5-Fold Cross-Validation
4. Trains final models on full training data
5. Evaluates on holdout test set
6. Computes SHAP explanations
7. Serializes all artifacts for Streamlit deployment
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import generate_cardiovascular_data, inject_missingness
from src.preprocessing import (
    ClinicalDataPreprocessor,
    engineer_features,
    TARGET_COL,
    TIME_COL,
)
from src.models import (
    build_xgboost_isotonic,
    build_logistic_regression_l2,
    SurvivalForestClassifier,
    EnsembleRiskModel,
)
from src.evaluation import stratified_cv_evaluate, generate_holdout_report
from src.explainability import SHAPExplainer

from sklearn.model_selection import train_test_split


def run_pipeline(
    data_path: str = None,
    output_dir: str = None,
    random_state: int = 42,
    n_samples: int = 12000,
    run_cv: bool = True,
    n_folds: int = 5,
) -> dict:
    """
    Execute the complete training pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to existing CSV data. If None, generates synthetic data.
    output_dir : str, optional
        Directory to save all artifacts. Defaults to project models/ dir.
    random_state : int
        Master random seed.
    n_samples : int
        Number of synthetic samples if generating data.
    run_cv : bool
        Whether to run cross-validation.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    dict
        Pipeline results including metrics and model paths.
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "models")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("+" + "=" * 68 + "+")
    print("|  CARDIOVASCULAR RISK STRATIFICATION ENGINE - TRAINING PIPELINE     |")
    print("+" + "=" * 68 + "+")

    # ==================================================================
    # STEP 1: Data Generation / Loading
    # ==================================================================
    print("\n[STEP 1] Data Preparation")

    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} records from {data_path}")
    else:
        print(f"  Generating {n_samples} synthetic patient records...")
        df = generate_cardiovascular_data(n_samples=n_samples, random_state=random_state)
        df = inject_missingness(df, missing_rate=0.03, random_state=random_state)
        csv_path = data_dir / "cardiovascular_risk_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}")

    print(f"  Shape: {df.shape}")
    print(f"  Event rate: {df[TARGET_COL].mean():.3f}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    # ==================================================================
    # STEP 2: Train/Test Split + Preprocessing
    # ==================================================================
    print("\n[STEP 2] Preprocessing & Feature Engineering")

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df[TARGET_COL]
    )
    print(f"  Train: {len(df_train)} | Test: {len(df_test)}")

    preprocessor = ClinicalDataPreprocessor()
    X_train, y_train, time_train = preprocessor.fit_transform(df_train)

    y_test = df_test[TARGET_COL].values.astype(int)
    time_test = df_test[TIME_COL].values.astype(float)
    X_test = preprocessor.transform(df_test)

    feature_names = preprocessor.feature_names
    print(f"  Features: {len(feature_names)}")
    print(f"  Feature list: {feature_names}")

    # Save preprocessor
    preprocessor.save(str(output_path / "preprocessor.joblib"))

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Stratified K-Fold Cross-Validation
    # ══════════════════════════════════════════════════════════════════
    cv_results = None
    if run_cv:
        print("\n[STEP 3] Stratified {}-Fold Cross-Validation".format(n_folds))
        cv_results = stratified_cv_evaluate(
            X_train, y_train, time_train,
            n_folds=n_folds, random_state=random_state,
        )
    else:
        print("\n[STEP 3] Cross-Validation (skipped)")

    # ==================================================================
    # STEP 4: Train Final Models on Full Training Data
    # ==================================================================
    print("\n[STEP 4] Training Final Models")

    # Model 1: Isotonic-Calibrated XGBoost
    print("  Training XGBoost with isotonic calibration...")
    xgb_model = build_xgboost_isotonic(random_state=random_state)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, str(output_path / "xgboost_isotonic.joblib"))
    print("  [OK] XGBoost-Isotonic saved")

    # Model 2: Random Survival Forest
    print("  Training Random Survival Forest...")
    rsf_model = SurvivalForestClassifier(random_state=random_state)
    rsf_model.fit(X_train, y_train, time=time_train)
    joblib.dump(rsf_model, str(output_path / "rsf_model.joblib"))
    print("  [OK] RSF saved")

    # Model 3: L2-Regularized Logistic Regression
    print("  Training L2-Regularized Logistic Regression...")
    lr_model = build_logistic_regression_l2(random_state=random_state)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, str(output_path / "logistic_l2.joblib"))
    print("  [OK] Logistic-L2 saved")

    # Ensemble
    ensemble = EnsembleRiskModel()
    ensemble.add_model("xgboost", xgb_model)
    ensemble.add_model("rsf", rsf_model)
    ensemble.add_model("logistic", lr_model)
    joblib.dump(ensemble, str(output_path / "ensemble_model.joblib"))
    print("  [OK] Ensemble saved")

    # ==================================================================
    # STEP 5: Holdout Test Evaluation
    # ==================================================================
    print("\n[STEP 5] Holdout Test Evaluation")

    predictions = {
        "XGBoost-Isotonic": xgb_model.predict_proba(X_test)[:, 1],
        "RSF": rsf_model.predict_proba(X_test)[:, 1],
        "Logistic-L2": lr_model.predict_proba(X_test)[:, 1],
        "Ensemble": ensemble.predict_proba(X_test)[:, 1],
    }

    holdout_report = generate_holdout_report(y_test, predictions)
    holdout_report.to_csv(str(output_path / "holdout_metrics.csv"))

    # ==================================================================
    # STEP 6: SHAP Explanations
    # ==================================================================
    print("\n[STEP 6] Computing SHAP Explanations")

    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # XGBoost SHAP (primary)
    print("  Computing SHAP values for XGBoost...")
    xgb_explainer = SHAPExplainer(
        xgb_model, feature_names, model_type="xgboost",
        background_data=X_train[:200],
    )
    xgb_shap_values = xgb_explainer.compute_shap_values(X_test, max_samples=500)

    # Global importance
    importance_df = xgb_explainer.get_global_importance()
    importance_df.to_csv(str(output_path / "feature_importance.csv"), index=False)
    print("\n  Global Feature Importance (Top 10):")
    print(importance_df.head(10).to_string(index=False))

    # Save plots
    try:
        xgb_explainer.plot_summary(X_test[:500], save_path=str(plots_dir / "shap_summary.png"))
        xgb_explainer.plot_waterfall(0, X_test[:500], save_path=str(plots_dir / "shap_waterfall_example.png"))
        print("  [OK] SHAP plots saved")
    except Exception as e:
        print(f"  [WARN] Plot generation warning: {e}")

    # Save SHAP values and explainer for Streamlit
    joblib.dump(xgb_shap_values, str(output_path / "shap_values.joblib"))
    joblib.dump(xgb_explainer, str(output_path / "shap_explainer.joblib"))

    # Save background data sample for inference
    np.save(str(output_path / "background_data.npy"), X_train[:200])

    # ==================================================================
    # STEP 7: Latency Benchmark
    # ==================================================================
    print("\n[STEP 7] Inference Latency Benchmark")

    single_patient = X_test[0:1]
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        ensemble.predict_proba(single_patient)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"  Single patient inference (100 runs):")
    print(f"    P50: {p50:.1f}ms | P95: {p95:.1f}ms | P99: {p99:.1f}ms")

    # ==================================================================
    # STEP 8: Save Pipeline Metadata
    # ==================================================================
    total_time = time.time() - start_time

    metadata = {
        "n_samples": n_samples,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "event_rate": float(df[TARGET_COL].mean()),
        "train_size": len(df_train),
        "test_size": len(df_test),
        "cv_folds": n_folds,
        "random_state": random_state,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "holdout_metrics": {
            name: {
                "roc_auc": float(holdout_report.loc[name, "roc_auc"]),
                "brier_score": float(holdout_report.loc[name, "brier_score"]),
            }
            for name in holdout_report.index
        },
        "total_training_time_seconds": round(total_time, 1),
    }

    if cv_results and "summary" in cv_results:
        metadata["cv_summary"] = {
            k: {mk: round(float(mv), 4) for mk, mv in v.items()}
            for k, v in cv_results["summary"].items()
        }

    with open(str(output_path / "pipeline_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE in {total_time:.1f}s")
    print(f"  Artifacts saved to: {output_path}")
    print(f"{'=' * 70}")

    return {
        "metadata": metadata,
        "models": {
            "xgboost": xgb_model,
            "rsf": rsf_model,
            "logistic": lr_model,
            "ensemble": ensemble,
        },
        "preprocessor": preprocessor,
        "cv_results": cv_results,
        "holdout_report": holdout_report,
        "shap_explainer": xgb_explainer,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    results = run_pipeline(
        output_dir=str(PROJECT_ROOT / "models"),
        n_samples=12000,
        run_cv=True,
        n_folds=5,
    )
