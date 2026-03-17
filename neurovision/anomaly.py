"""
Anomaly Detector
Uses Isolation Forest and Local Outlier Factor for anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Detects anomalies in datasets using:
    - Isolation Forest (default)
    - Local Outlier Factor
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Args:
            contamination: Expected fraction of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=False
        )
        self.scaler = StandardScaler()

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detect anomalies using Isolation Forest on numeric columns.

        Args:
            df: Input DataFrame (without target)

        Returns:
            dict with anomaly flags, scores, and indices
        """
        # Use only numeric columns
        num_df = df.select_dtypes(include=[np.number]).copy()
        num_df = num_df.fillna(num_df.median())

        if num_df.empty or len(num_df) < 10:
            return {"n_anomalies": 0, "anomaly_indices": [], "scores": []}

        X_scaled = self.scaler.fit_transform(num_df)

        # Isolation Forest
        iso_preds = self.iso_forest.fit_predict(X_scaled)  # -1 = anomaly
        iso_scores = self.iso_forest.score_samples(X_scaled)

        # LOF
        lof_preds = self.lof.fit_predict(X_scaled)  # -1 = anomaly

        # Combine: flag if either method flags it
        combined = ((iso_preds == -1) | (lof_preds == -1))
        anomaly_indices = np.where(combined)[0].tolist()

        return {
            "n_anomalies": int(combined.sum()),
            "anomaly_indices": anomaly_indices,
            "iso_scores": iso_scores.tolist(),
            "iso_predictions": iso_preds.tolist(),
            "lof_predictions": lof_preds.tolist(),
            "anomaly_flags": combined.tolist(),
            "anomaly_fraction": float(combined.mean()),
        }

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return anomaly scores for each row.
        Lower = more anomalous.

        Args:
            df: Input DataFrame

        Returns:
            Array of anomaly scores
        """
        num_df = df.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.transform(num_df)
        return self.iso_forest.score_samples(X_scaled)
