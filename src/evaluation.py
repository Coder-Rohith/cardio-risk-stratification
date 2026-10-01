"""
Model Evaluation & Cross-Validation
=====================================
Implements Stratified 5-Fold Cross-Validation with:
- ROC-AUC (discriminative performance)
- Brier Score (calibration quality)
- Precision, Recall, F1-Score
- Calibration curve analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_curve,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from typing import Dict, List, Tuple, Any
import time
import warnings

from src.models import (
    build_xgboost_isotonic,
    build_logistic_regression_l2,
    SurvivalForestClassifier,
    EnsembleRiskModel,
)

warnings.filterwarnings("ignore")


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Name for display.

    Returns
    -------
    Dict with metric names and values.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    return metrics


def stratified_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    time_to_event: np.ndarray = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Stratified K-Fold Cross-Validation for all three models + ensemble.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary target.
    time_to_event : np.ndarray, optional
        Time-to-event for survival forest.
    n_folds : int
        Number of CV folds.
    random_state : int
        Random seed.

    Returns
    -------
    Dict containing per-fold and aggregated metrics for each model.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {
        "xgboost": {"roc_auc": [], "brier_score": [], "fold_metrics": []},
        "rsf": {"roc_auc": [], "brier_score": [], "fold_metrics": []},
        "logistic": {"roc_auc": [], "brier_score": [], "fold_metrics": []},
        "ensemble": {"roc_auc": [], "brier_score": [], "fold_metrics": []},
    }

    fold_times = []

    print("\n" + "=" * 70)
    print("  STRATIFIED {}-FOLD CROSS-VALIDATION".format(n_folds))
    print("=" * 70)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        time_train = time_to_event[train_idx] if time_to_event is not None else None
        time_val = time_to_event[val_idx] if time_to_event is not None else None

        print(f"\n  -- Fold {fold_idx + 1}/{n_folds} -------------------------")
        print(f"     Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"     Event rate - Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")

        # -- 1. Isotonic-Calibrated XGBoost ----------------------------
        xgb_model = build_xgboost_isotonic(random_state=random_state)
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
        xgb_metrics = evaluate_predictions(y_val, xgb_prob, "XGBoost-Isotonic")

        # -- 2. Random Survival Forest ---------------------------------
        rsf_model = SurvivalForestClassifier(random_state=random_state)
        rsf_model.fit(X_train, y_train, time=time_train)
        rsf_prob = rsf_model.predict_proba(X_val)[:, 1]
        rsf_metrics = evaluate_predictions(y_val, rsf_prob, "RSF")

        # -- 3. L2-Regularized Logistic Regression ---------------------
        lr_model = build_logistic_regression_l2(random_state=random_state)
        lr_model.fit(X_train, y_train)
        lr_prob = lr_model.predict_proba(X_val)[:, 1]
        lr_metrics = evaluate_predictions(y_val, lr_prob, "Logistic-L2")

        # -- 4. Ensemble ----------------------------------------------
        ensemble = EnsembleRiskModel()
        ensemble.add_model("xgboost", xgb_model)
        ensemble.add_model("rsf", rsf_model)
        ensemble.add_model("logistic", lr_model)
        ens_prob = ensemble.predict_proba(X_val)[:, 1]
        ens_metrics = evaluate_predictions(y_val, ens_prob, "Ensemble")

        # Store results
        for name, metrics in [
            ("xgboost", xgb_metrics),
            ("rsf", rsf_metrics),
            ("logistic", lr_metrics),
            ("ensemble", ens_metrics),
        ]:
            results[name]["roc_auc"].append(metrics["roc_auc"])
            results[name]["brier_score"].append(metrics["brier_score"])
            results[name]["fold_metrics"].append(metrics)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)

        print(f"\n     {'Model':<22} {'ROC-AUC':>8} {'Brier':>8}")
        print(f"     {'-' * 40}")
        print(f"     {'XGBoost-Isotonic':<22} {xgb_metrics['roc_auc']:>8.4f} {xgb_metrics['brier_score']:>8.4f}")
        print(f"     {'RSF':<22} {rsf_metrics['roc_auc']:>8.4f} {rsf_metrics['brier_score']:>8.4f}")
        print(f"     {'Logistic-L2':<22} {lr_metrics['roc_auc']:>8.4f} {lr_metrics['brier_score']:>8.4f}")
        print(f"     {'Ensemble':<22} {ens_metrics['roc_auc']:>8.4f} {ens_metrics['brier_score']:>8.4f}")
        print(f"     Fold time: {fold_time:.1f}s")

    # -- Aggregate Results ---------------------------------------------
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS (mean +/- std)")
    print("=" * 70)
    print(f"\n  {'Model':<22} {'ROC-AUC':>16} {'Brier Score':>16}")
    print(f"  {'-' * 56}")

    summary = {}
    for name in ["xgboost", "rsf", "logistic", "ensemble"]:
        auc_mean = np.mean(results[name]["roc_auc"])
        auc_std = np.std(results[name]["roc_auc"])
        brier_mean = np.mean(results[name]["brier_score"])
        brier_std = np.std(results[name]["brier_score"])

        display_name = {
            "xgboost": "XGBoost-Isotonic",
            "rsf": "RSF",
            "logistic": "Logistic-L2",
            "ensemble": "Ensemble",
        }[name]

        print(
            f"  {display_name:<22} "
            f"{auc_mean:.4f} +/- {auc_std:.4f}  "
            f"{brier_mean:.4f} +/- {brier_std:.4f}"
        )

        summary[name] = {
            "roc_auc_mean": auc_mean,
            "roc_auc_std": auc_std,
            "brier_mean": brier_mean,
            "brier_std": brier_std,
        }

    print(f"\n  Mean fold time: {np.mean(fold_times):.1f}s")
    print("=" * 70)

    results["summary"] = summary
    results["fold_times"] = fold_times

    return results


def generate_holdout_report(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Generate a detailed evaluation report on holdout test data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    predictions : Dict[str, np.ndarray]
        Model name → predicted probabilities.

    Returns
    -------
    pd.DataFrame
        Metrics comparison table.
    """
    rows = []
    for name, y_prob in predictions.items():
        metrics = evaluate_predictions(y_true, y_prob, name)
        rows.append(metrics)

    report_df = pd.DataFrame(rows).set_index("model")
    report_df = report_df.round(4)

    print("\n" + "=" * 70)
    print("  HOLDOUT TEST SET EVALUATION")
    print("=" * 70)
    print(report_df.to_string())
    print("=" * 70)

    return report_df
