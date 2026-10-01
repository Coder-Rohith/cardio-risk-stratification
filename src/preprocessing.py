"""
Data Preprocessing Pipeline
============================
Handles clinical data preprocessing including:
- Missing value imputation (median for continuous, mode for categorical)
- Feature engineering (derived clinical ratios & risk indicators)
- Standard scaling for continuous features
- Train/test splitting with stratification
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import joblib
from pathlib import Path


# ── Feature Classification ────────────────────────────────────────────
CONTINUOUS_FEATURES = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate", "bmi",
    "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol",
    "triglycerides", "fasting_glucose", "hba1c", "crp_level", "creatinine",
]

CATEGORICAL_FEATURES = [
    "sex", "smoking_status", "physical_activity",
    "alcohol_consumption", "diabetes", "hypertension_med", "family_history_cvd",
]

TARGET_COL = "cvd_event"
TIME_COL = "time_to_event_days"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer clinically-meaningful derived features.

    Creates composite risk indicators and ratio features used in
    cardiovascular risk scoring systems (Framingham, ASCVD).

    Parameters
    ----------
    df : pd.DataFrame
        Raw patient data.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features appended.
    """
    df = df.copy()

    # ── Lipid Ratios ──────────────────────────────────────────────────
    df["chol_hdl_ratio"] = df["total_cholesterol"] / (df["hdl_cholesterol"] + 1)
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1)

    # ── Blood Pressure Metrics ────────────────────────────────────────
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = (
        df["diastolic_bp"] + (df["pulse_pressure"] / 3)
    )

    # ── Metabolic Risk Score ──────────────────────────────────────────
    df["metabolic_risk"] = (
        (df["bmi"] > 30).astype(int)
        + (df["fasting_glucose"] > 100).astype(int)
        + (df["triglycerides"] > 150).astype(int)
        + (df["hdl_cholesterol"] < 40).astype(int)
        + (df["systolic_bp"] > 130).astype(int)
    )

    # ── Age-Risk Interactions ─────────────────────────────────────────
    df["age_sbp_product"] = df["age"] * df["systolic_bp"] / 100
    df["age_chol_product"] = df["age"] * df["total_cholesterol"] / 1000

    # ── BMI Categories ────────────────────────────────────────────────
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # ── BP Stage ──────────────────────────────────────────────────────
    conditions = [
        df["systolic_bp"] < 120,
        (df["systolic_bp"] >= 120) & (df["systolic_bp"] < 130),
        (df["systolic_bp"] >= 130) & (df["systolic_bp"] < 140),
        df["systolic_bp"] >= 140,
    ]
    df["bp_stage"] = np.select(conditions, [0, 1, 2, 3], default=0)

    return df


# ── Derived Feature Lists ────────────────────────────────────────────
ENGINEERED_CONTINUOUS = [
    "chol_hdl_ratio", "ldl_hdl_ratio", "tg_hdl_ratio",
    "pulse_pressure", "mean_arterial_pressure",
    "age_sbp_product", "age_chol_product",
]

ENGINEERED_CATEGORICAL = [
    "metabolic_risk", "bmi_category", "bp_stage",
]

ALL_FEATURES = (
    CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    + ENGINEERED_CONTINUOUS + ENGINEERED_CATEGORICAL
)


class ClinicalDataPreprocessor:
    """
    End-to-end clinical data preprocessor with fit/transform semantics.

    Handles imputation, feature engineering, and scaling in a single
    pipeline that can be serialized for inference deployment.
    """

    def __init__(self):
        self.continuous_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.feature_names: List[str] = []

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the preprocessor on training data and return transformed arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Raw training data with target column.

        Returns
        -------
        Tuple of (X, y, time_to_event)
        """
        # Engineer features first
        df = engineer_features(df)

        # Separate targets
        y = df[TARGET_COL].values.astype(int)
        time = df[TIME_COL].values.astype(float) if TIME_COL in df.columns else None

        # Impute missing values
        cont_cols = [c for c in CONTINUOUS_FEATURES + ENGINEERED_CONTINUOUS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL if c in df.columns]

        df[cont_cols] = self.continuous_imputer.fit_transform(df[cont_cols])
        df[cat_cols] = self.categorical_imputer.fit_transform(df[cat_cols])

        # Scale continuous features
        df[cont_cols] = self.scaler.fit_transform(df[cont_cols])

        self.feature_names = cont_cols + cat_cols
        X = df[self.feature_names].values.astype(np.float32)

        self._is_fitted = True
        return X, y, time

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data to transform.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit before transform.")

        df = engineer_features(df)

        cont_cols = [c for c in CONTINUOUS_FEATURES + ENGINEERED_CONTINUOUS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL if c in df.columns]

        df[cont_cols] = self.continuous_imputer.transform(df[cont_cols])
        df[cat_cols] = self.categorical_imputer.transform(df[cat_cols])
        df[cont_cols] = self.scaler.transform(df[cont_cols])

        return df[self.feature_names].values.astype(np.float32)

    def save(self, path: str) -> None:
        """Save preprocessor state to disk."""
        joblib.dump(
            {
                "continuous_imputer": self.continuous_imputer,
                "categorical_imputer": self.categorical_imputer,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            },
            path,
        )
        print(f"[PREPROC] Saved preprocessor -> {path}")

    def load(self, path: str) -> "ClinicalDataPreprocessor":
        """Load preprocessor state from disk."""
        state = joblib.load(path)
        self.continuous_imputer = state["continuous_imputer"]
        self.categorical_imputer = state["categorical_imputer"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self._is_fitted = True
        return self


def prepare_data(
    data_path: str, test_size: float = 0.2, random_state: int = 42
) -> Dict:
    """
    Load and prepare data for model training.

    Parameters
    ----------
    data_path : str
        Path to CSV data file.
    test_size : float
        Proportion for holdout test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict with keys: X_train, X_test, y_train, y_test,
                     time_train, time_test, preprocessor, feature_names
    """
    df = pd.read_csv(data_path)
    print(f"[PREPROC] Loaded {len(df)} records with {df.shape[1]} columns")
    print(f"[PREPROC] Event rate: {df[TARGET_COL].mean():.3f}")
    print(f"[PREPROC] Missing values: {df.isnull().sum().sum()}")

    # Split before preprocessing to prevent leakage
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[TARGET_COL],
    )

    preprocessor = ClinicalDataPreprocessor()
    X_train, y_train, time_train = preprocessor.fit_transform(df_train)
    X_test, y_test, time_test = preprocessor.fit_transform(df_test)

    # Re-fit on train only, then transform test
    preprocessor2 = ClinicalDataPreprocessor()
    X_train, y_train, time_train = preprocessor2.fit_transform(df_train)

    # For test, we need targets separately
    y_test = df_test[TARGET_COL].values.astype(int)
    time_test = df_test[TIME_COL].values.astype(float) if TIME_COL in df_test.columns else None
    X_test = preprocessor2.transform(df_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "time_train": time_train,
        "time_test": time_test,
        "preprocessor": preprocessor2,
        "feature_names": preprocessor2.feature_names,
        "df_train": df_train,
        "df_test": df_test,
    }
