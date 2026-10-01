"""
SHAP Explainability Module
============================
Implements SHAP TreeExplainer for XGBoost and KernelExplainer fallback
for model-agnostic explanations. Provides:

- Global feature importance (mean absolute SHAP values)
- Local explanations (individual patient force/waterfall plots)
- Feature interaction analysis
- Exportable explanation summaries for clinician reports
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import warnings
import io

warnings.filterwarnings("ignore")


class SHAPExplainer:
    """
    SHAP-based model explainability for cardiovascular risk models.

    Uses TreeExplainer for tree-based models (XGBoost, RSF) and
    KernelExplainer as a fallback for other model types.

    Parameters
    ----------
    model : fitted model
        The trained model to explain.
    feature_names : List[str]
        Feature names for interpretability.
    model_type : str
        One of 'xgboost', 'rsf', 'logistic', 'ensemble'.
    background_data : np.ndarray, optional
        Background dataset for KernelExplainer (subsample for speed).
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        model_type: str = "xgboost",
        background_data: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

        self._init_explainer(background_data)

    def _init_explainer(self, background_data: Optional[np.ndarray] = None):
        """Initialize the appropriate SHAP explainer."""
        if self.model_type == "xgboost":
            # For CalibratedClassifierCV wrapping XGBoost, extract base estimator
            try:
                if hasattr(self.model, "calibrated_classifiers_"):
                    # Use the first calibrated classifier's base estimator
                    base = self.model.calibrated_classifiers_[0].estimator
                    self.explainer = shap.TreeExplainer(base)
                elif hasattr(self.model, "estimator"):
                    self.explainer = shap.TreeExplainer(self.model.estimator)
                else:
                    self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                if background_data is not None:
                    bg = shap.sample(background_data, min(100, len(background_data)))
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, bg
                    )

        elif self.model_type == "rsf":
            try:
                if hasattr(self.model, "rsf_") and hasattr(self.model.rsf_, "estimators_"):
                    self.explainer = shap.TreeExplainer(self.model.rsf_)
                else:
                    if background_data is not None:
                        bg = shap.sample(background_data, min(100, len(background_data)))
                        self.explainer = shap.KernelExplainer(
                            lambda x: self.model.predict_proba(x)[:, 1], bg
                        )
            except Exception:
                if background_data is not None:
                    bg = shap.sample(background_data, min(100, len(background_data)))
                    self.explainer = shap.KernelExplainer(
                        lambda x: self.model.predict_proba(x)[:, 1], bg
                    )

        elif self.model_type == "logistic":
            if background_data is not None:
                bg = shap.sample(background_data, min(100, len(background_data)))
                self.explainer = shap.LinearExplainer(
                    self.model, bg
                )

        else:
            if background_data is not None:
                bg = shap.sample(background_data, min(100, len(background_data)))
                self.explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1], bg
                )

    def compute_shap_values(
        self, X: np.ndarray, max_samples: int = 500
    ) -> np.ndarray:
        """
        Compute SHAP values for a dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        max_samples : int
            Maximum samples to explain (for performance).

        Returns
        -------
        np.ndarray
            SHAP values matrix.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Provide background data.")

        X_subset = X[:max_samples] if len(X) > max_samples else X

        shap_vals = self.explainer.shap_values(X_subset)

        # Handle multi-output (binary classification returns list of 2)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class

        self.shap_values = shap_vals
        if hasattr(self.explainer, "expected_value"):
            ev = self.explainer.expected_value
            self.expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev
        else:
            self.expected_value = 0.0

        return shap_vals

    def get_global_importance(self, X: np.ndarray = None) -> pd.DataFrame:
        """
        Compute global feature importance (mean |SHAP|).

        Parameters
        ----------
        X : np.ndarray, optional
            If SHAP values not yet computed, compute them on this data.

        Returns
        -------
        pd.DataFrame
            Feature importance ranked by mean absolute SHAP value.
        """
        if self.shap_values is None:
            if X is None:
                raise ValueError("Must provide X or compute SHAP values first.")
            self.compute_shap_values(X)

        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["pct_contribution"] = (
            importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum() * 100
        ).round(2)

        return importance_df

    def explain_patient(
        self, patient_features: np.ndarray, X_background: np.ndarray = None
    ) -> Dict:
        """
        Generate a local explanation for a single patient.

        Parameters
        ----------
        patient_features : np.ndarray
            1D or 2D array of single patient's features.
        X_background : np.ndarray, optional
            Background data for context.

        Returns
        -------
        Dict with patient SHAP values and interpretation.
        """
        if patient_features.ndim == 1:
            patient_features = patient_features.reshape(1, -1)

        if self.explainer is None:
            raise RuntimeError("Explainer not initialized.")

        shap_vals = self.explainer.shap_values(patient_features)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_vals = shap_vals.flatten()

        # Build feature-level explanation
        explanations = []
        for i, (name, sv) in enumerate(zip(self.feature_names, shap_vals)):
            explanations.append({
                "feature": name,
                "value": float(patient_features[0, i]),
                "shap_value": float(sv),
                "direction": "increases risk" if sv > 0 else "decreases risk",
                "magnitude": abs(float(sv)),
            })

        explanations.sort(key=lambda x: x["magnitude"], reverse=True)

        ev = self.expected_value if self.expected_value is not None else 0.0

        return {
            "shap_values": shap_vals,
            "expected_value": float(ev),
            "explanations": explanations,
            "top_risk_factors": [e for e in explanations[:5] if e["shap_value"] > 0],
            "top_protective_factors": [e for e in explanations[:5] if e["shap_value"] < 0],
        }

    def plot_summary(
        self, X: np.ndarray, save_path: Optional[str] = None, max_display: int = 15
    ) -> Optional[plt.Figure]:
        """Generate SHAP summary beeswarm plot."""
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_display = X[:len(self.shap_values)]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X_display,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            fig = plt.gcf()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Summary plot saved -> {save_path}")

        return plt.gcf()

    def plot_waterfall(
        self, patient_idx: int, X: np.ndarray, save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """Generate SHAP waterfall plot for a single patient."""
        if self.shap_values is None:
            self.compute_shap_values(X)

        ev = self.expected_value if self.expected_value is not None else 0.0

        explanation = shap.Explanation(
            values=self.shap_values[patient_idx],
            base_values=ev,
            data=X[patient_idx],
            feature_names=self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()

        if save_path:
            fig = plt.gcf()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Waterfall plot saved -> {save_path}")

        return plt.gcf()

    def plot_force(
        self, patient_idx: int, X: np.ndarray
    ) -> Optional[object]:
        """Generate SHAP force plot for a single patient."""
        if self.shap_values is None:
            self.compute_shap_values(X)

        ev = self.expected_value if self.expected_value is not None else 0.0

        force_plot = shap.force_plot(
            ev,
            self.shap_values[patient_idx],
            X[patient_idx],
            feature_names=self.feature_names,
            matplotlib=False,
        )

        return force_plot

    def get_force_plot_html(self, patient_idx: int, X: np.ndarray) -> str:
        """Get SHAP force plot as HTML string for embedding in Streamlit."""
        if self.shap_values is None:
            self.compute_shap_values(X)

        ev = self.expected_value if self.expected_value is not None else 0.0

        force_plot = shap.force_plot(
            ev,
            self.shap_values[patient_idx],
            X[patient_idx],
            feature_names=self.feature_names,
            matplotlib=False,
        )

        return shap.getjs() + force_plot.html()
