"""
Model Trainer
Trains multiple ML models in parallel and returns a ranked leaderboard.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from joblib import Parallel, delayed
from tqdm import tqdm


class ModelTrainer:
    """
    Trains all supported ML models in parallel and returns a leaderboard
    sorted by the primary metric.
    """

    CLASSIFICATION_MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    REGRESSION_MODELS = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    }

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        test_size: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Train all models and return sorted leaderboard.

        Args:
            X: Feature matrix
            y: Target vector
            problem_type: 'classification' or 'regression'
            test_size: Test split fraction

        Returns:
            List of dicts sorted by best metric (desc for classification, asc for RMSE)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        models = (
            self.CLASSIFICATION_MODELS
            if problem_type == "classification"
            else self.REGRESSION_MODELS
        )

        results = []
        for name, model in tqdm(models.items(), desc="Training models"):
            try:
                entry = self._train_single(
                    name, model, X_train, X_test, y_train, y_test, problem_type
                )
                entry["X_test"] = X_test
                results.append(entry)
            except Exception as e:
                print(f"  [WARN] {name} failed: {e}")

        # Sort by primary metric
        if problem_type == "classification":
            results.sort(key=lambda x: x.get("f1", 0), reverse=True)
        else:
            results.sort(key=lambda x: x.get("rmse", float("inf")))

        return results

    def _train_single(
        self,
        name: str,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str,
    ) -> Dict[str, Any]:
        """Train a single model and evaluate metrics."""
        import copy
        m = copy.deepcopy(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        entry = {"model_name": name, "model": m}

        if problem_type == "classification":
            entry["accuracy"] = accuracy_score(y_test, y_pred)
            entry["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            try:
                if hasattr(m, "predict_proba"):
                    y_prob = m.predict_proba(X_test)
                    n_classes = len(np.unique(y_test))
                    if n_classes == 2:
                        entry["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        entry["roc_auc"] = roc_auc_score(
                            y_test, y_prob, multi_class="ovr", average="weighted"
                        )
                else:
                    entry["roc_auc"] = 0.0
            except Exception:
                entry["roc_auc"] = 0.0
        else:
            entry["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
            entry["mae"] = mean_absolute_error(y_test, y_pred)
            entry["r2"] = r2_score(y_test, y_pred)

        return entry
