"""
Model Definitions & Training
==============================
Three-model cardiovascular risk stratification ensemble:

1. Isotonic-Calibrated XGBoost
   - Gradient-boosted trees with isotonic regression calibration
   - Primary discriminative model

2. Random Survival Forest (adapted for binary classification)
   - scikit-survival RandomSurvivalForest
   - Provides survival-aware risk scores

3. L2-Regularized Logistic Regression
   - Ridge-penalized linear model
   - Baseline calibrated estimator
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def build_xgboost_isotonic(random_state: int = 42) -> CalibratedClassifierCV:
    """
    Build an XGBoost classifier with isotonic calibration.

    Isotonic calibration fits a non-parametric monotonic function to
    map raw XGBoost scores to well-calibrated probabilities, which is
    critical for clinical risk thresholds.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    CalibratedClassifierCV
        Isotonic-calibrated XGBoost classifier.
    """
    base_xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    calibrated_xgb = CalibratedClassifierCV(
        estimator=base_xgb,
        method="isotonic",
        cv=3,
    )

    return calibrated_xgb


def build_logistic_regression_l2(random_state: int = 42) -> LogisticRegression:
    """
    Build an L2-regularized (Ridge) logistic regression model.

    L2 regularization prevents coefficient explosion and provides
    a well-calibrated baseline that is inherently interpretable.

    Parameters
    ----------
    random_state : int
        Random seed.

    Returns
    -------
    LogisticRegression
        L2-regularized logistic regression.
    """
    return LogisticRegression(
        penalty="l2",
        C=0.5,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )


class SurvivalForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Adapter wrapping RandomSurvivalForest for binary classification.

    Extracts cumulative hazard function at a specified time horizon
    and converts to risk probability for the binary classification task.

    Parameters
    ----------
    n_estimators : int
        Number of survival trees.
    max_depth : int or None
        Maximum tree depth.
    min_samples_split : int
        Minimum samples required to split.
    min_samples_leaf : int
        Minimum samples in leaf nodes.
    time_horizon : float
        Time point (in days) at which to evaluate survival probability.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        time_horizon: float = 1825.0,  # 5 years in days
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.time_horizon = time_horizon
        self.random_state = random_state
        self.rsf_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, time=None):
        """
        Fit the Random Survival Forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary event indicator.
        time : array-like of shape (n_samples,), optional
            Time-to-event. If None, synthetic times are generated.
        """
        try:
            from sksurv.ensemble import RandomSurvivalForest

            # Create structured array for sksurv
            if time is None:
                rng = np.random.RandomState(self.random_state)
                time = np.where(
                    y == 1,
                    rng.exponential(1000, len(y)),
                    rng.exponential(2000, len(y)),
                ).clip(30, 3650)

            y_surv = np.array(
                [(bool(e), float(t)) for e, t in zip(y, time)],
                dtype=[("event", bool), ("time", float)],
            )

            self.rsf_ = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.rsf_.fit(X, y_surv)
            self._use_sksurv = True

        except ImportError:
            # Fallback: use XGBoost with survival-like approach
            print("[WARN] scikit-survival not available, using XGBoost fallback for RSF")
            from xgboost import XGBClassifier
            self.rsf_ = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
            )
            self.rsf_.fit(X, y)
            self._use_sksurv = False

        return self

    def predict_proba(self, X):
        """
        Predict risk probabilities.

        For sksurv: extracts 1 - survival_function(time_horizon).
        For fallback: uses XGBoost predict_proba directly.
        """
        if self._use_sksurv:
            surv_funcs = self.rsf_.predict_survival_function(X)
            risk_scores = np.array([
                1 - fn(self.time_horizon) if self.time_horizon <= fn.x[-1]
                else 1 - fn(fn.x[-1])
                for fn in surv_funcs
            ])
            risk_scores = risk_scores.clip(0, 1)
            return np.column_stack([1 - risk_scores, risk_scores])
        else:
            return self.rsf_.predict_proba(X)

    def predict(self, X):
        """Predict binary class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class EnsembleRiskModel(BaseEstimator, ClassifierMixin):
    """
    Ensemble meta-model combining all three risk estimators.

    Computes a weighted average of calibrated probabilities from
    each constituent model. Weights are optimized on validation
    data to minimize Brier score.

    Parameters
    ----------
    weights : Dict[str, float], optional
        Model weights. If None, uses uniform weights.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "xgboost": 0.50,
            "rsf": 0.25,
            "logistic": 0.25,
        }
        self.models: Dict[str, BaseEstimator] = {}
        self.classes_ = np.array([0, 1])

    def add_model(self, name: str, model: BaseEstimator) -> None:
        """Register a fitted model component."""
        self.models[name] = model

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict weighted ensemble probabilities.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns: [P(no event), P(event)]
        """
        total_weight = sum(self.weights[k] for k in self.models)
        ensemble_proba = np.zeros((X.shape[0], 2))

        for name, model in self.models.items():
            w = self.weights.get(name, 1.0 / len(self.models))
            proba = model.predict_proba(X)
            ensemble_proba += w * proba

        ensemble_proba /= total_weight
        return ensemble_proba

    def predict(self, X) -> np.ndarray:
        """Predict binary class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_individual_predictions(self, X) -> Dict[str, np.ndarray]:
        """Get risk probabilities from each individual model."""
        return {
            name: model.predict_proba(X)[:, 1]
            for name, model in self.models.items()
        }
