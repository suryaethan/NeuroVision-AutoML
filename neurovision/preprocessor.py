"""
Smart Preprocessor
Handles automatic data cleaning, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer


class SmartPreprocessor:
    """
    Automatically detects and handles:
    - Missing values (numerical + categorical)
    - Categorical encoding
    - Feature scaling
    - Problem type detection (classification vs regression)
    - DateTime feature extraction
    - Polynomial and interaction features
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.detected_problem_type = None
        self.label_encoders = {}
        self.scaler = None
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.target_encoder = LabelEncoder()
        self.feature_names_ = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full preprocessing pipeline: clean, encode, scale, engineer features.

        Args:
            df: Raw input DataFrame
            target_col: Target column name

        Returns:
            Tuple of (X_processed, y)
        """
        df = df.copy()

        # Separate target
        y_raw = df[target_col].copy()
        X = df.drop(columns=[target_col])

        # Detect problem type
        self.detected_problem_type = self._detect_problem_type(y_raw)

        # Encode target
        if self.detected_problem_type == "classification":
            y = pd.Series(
                self.target_encoder.fit_transform(y_raw.astype(str)),
                name=target_col
            )
        else:
            y = y_raw.astype(float)

        # Extract datetime features
        X = self._extract_datetime_features(X)

        # Separate numeric and categorical
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Impute
        if num_cols:
            X[num_cols] = self.num_imputer.fit_transform(X[num_cols])
        if cat_cols:
            X[cat_cols] = self.cat_imputer.fit_transform(X[cat_cols])

        # Encode categoricals
        X = self._encode_categoricals(X, cat_cols, fit=True)

        # Feature engineering
        X = self._engineer_features(X)

        # Scale
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        self.feature_names_ = list(X_scaled.columns)
        self._fitted = True
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")

        df = df.copy()
        df = self._extract_datetime_features(df)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if num_cols:
            df[num_cols] = self.num_imputer.transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.cat_imputer.transform(df[cat_cols])

        df = self._encode_categoricals(df, cat_cols, fit=False)
        df = self._engineer_features(df)

        # Align columns
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names_]

        return pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns,
            index=df.index
        )

    def _detect_problem_type(self, y: pd.Series) -> str:
        """Auto-detect whether this is classification or regression."""
        if y.dtype == "object" or y.dtype.name == "category":
            return "classification"
        unique_ratio = y.nunique() / len(y)
        if y.nunique() <= 20 or unique_ratio < 0.05:
            return "classification"
        return "regression"

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year, month, day, hour, dayofweek from datetime columns."""
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    dt = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    if dt.notna().sum() > len(df) * 0.5:
                        df[f"{col}_year"] = dt.dt.year
                        df[f"{col}_month"] = dt.dt.month
                        df[f"{col}_day"] = dt.dt.day
                        df[f"{col}_dayofweek"] = dt.dt.dayofweek
                        df[f"{col}_hour"] = dt.dt.hour
                        df.drop(columns=[col], inplace=True)
                except Exception:
                    pass
        return df

    def _encode_categoricals(self, df: pd.DataFrame, cat_cols: list, fit: bool) -> pd.DataFrame:
        """Encode categorical columns with Label Encoding."""
        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between top numeric columns."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Add ratio features for top-5 pairs
        top_cols = num_cols[:5]
        for i, c1 in enumerate(top_cols):
            for c2 in top_cols[i + 1:]:
                col_name = f"{c1}_x_{c2}"
                df[col_name] = df[c1] * df[c2]
        return df
