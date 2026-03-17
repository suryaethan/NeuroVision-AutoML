"""
SHAP Explainer
Generates SHAP-based explanations for any trained ML model.
"""

import numpy as np
import pandas as pd
from typing import Any, Optional
import shap
import warnings
warnings.filterwarnings("ignore")


class SHAPExplainer:
    """
    Wraps SHAP to provide:
    - Global feature importance (bar + beeswarm)
    - Per-prediction explanation (waterfall)
    - SHAP values for full test set
    """

    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.feature_names = None

    def explain(
        self,
        model: Any,
        X: pd.DataFrame,
        problem_type: str = "classification",
        max_samples: int = 200,
    ) -> dict:
        """
        Generate SHAP explanations for the model.

        Args:
            model: Trained sklearn-compatible model
            X: Feature DataFrame (test set)
            problem_type: 'classification' or 'regression'
            max_samples: Max rows to explain (for speed)

        Returns:
            dict with shap_values, expected_value, feature_names
        """
        self.feature_names = list(X.columns)
        X_sample = X.iloc[:max_samples].copy()

        try:
            # Try TreeExplainer first (fastest for tree-based models)
            self.explainer = shap.TreeExplainer(model)
            shap_vals = self.explainer.shap_values(X_sample)
            expected_value = self.explainer.expected_value
        except Exception:
            try:
                # Fallback to LinearExplainer
                self.explainer = shap.LinearExplainer(model, X_sample)
                shap_vals = self.explainer.shap_values(X_sample)
                expected_value = self.explainer.expected_value
            except Exception:
                # Final fallback: KernelExplainer (model-agnostic, slower)
                background = shap.sample(X_sample, min(50, len(X_sample)))
                self.explainer = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                    background
                )
                shap_vals = self.explainer.shap_values(X_sample)
                expected_value = self.explainer.expected_value

        # For multiclass, pick class 1 values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        self.shap_values = shap_vals

        # Compute feature importance
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)

        return {
            "shap_values": shap_vals,
            "expected_value": expected_value,
            "feature_names": self.feature_names,
            "importance_df": importance_df,
            "X_sample": X_sample,
        }

    def explain_single(
        self,
        model: Any,
        X_row: pd.DataFrame,
    ) -> dict:
        """
        Explain a single prediction.

        Args:
            model: Trained model (already has explainer)
            X_row: Single-row DataFrame

        Returns:
            dict with shap_values for this row
        """
        if self.explainer is None:
            raise RuntimeError("Call explain() first before explain_single().")

        row_shap = self.explainer.shap_values(X_row)
        if isinstance(row_shap, list):
            row_shap = row_shap[1] if len(row_shap) > 1 else row_shap[0]

        return {
            "shap_values": row_shap,
            "feature_names": self.feature_names,
            "feature_values": X_row.values[0],
        }

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Return top N most important features by mean |SHAP| value."""
        if self.shap_values is None:
            raise RuntimeError("Run explain() first.")
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        return df.head(n)
