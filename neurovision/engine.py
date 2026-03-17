"""
NeuroVision AutoML Engine
Core orchestration engine that runs the full autonomous ML pipeline.
"""

import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from neurovision.preprocessor import SmartPreprocessor
from neurovision.trainer import ModelTrainer
from neurovision.explainer import SHAPExplainer
from neurovision.anomaly import AnomalyDetector

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuroVisionEngine:
    """
    Core AutoML Engine that orchestrates the full ML pipeline.

    Usage:
        engine = NeuroVisionEngine()
        results = engine.run(data_path="data.csv", target_col="target")
    """

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.preprocessor = SmartPreprocessor(random_state=random_state)
        self.trainer = ModelTrainer(random_state=random_state, n_jobs=n_jobs)
        self.explainer = SHAPExplainer()
        self.anomaly_detector = AnomalyDetector(random_state=random_state)
        self.results = {}

    def run(
        self,
        data_path: str,
        target_col: str,
        problem_type: Optional[str] = None,
        test_size: float = 0.2,
        detect_anomalies: bool = True,
        explain: bool = True,
    ) -> dict:
        """
        Run the full autonomous AutoML pipeline.

        Args:
            data_path: Path to CSV dataset
            target_col: Name of the target column
            problem_type: 'classification' or 'regression' (auto-detected if None)
            test_size: Fraction of data for testing
            detect_anomalies: Run anomaly detection on data
            explain: Generate SHAP explanations for best model

        Returns:
            dict: Full results including best model, metrics, and SHAP values
        """
        start_time = time.time()

        # Banner
        console.print(Panel.fit(
            "[bold cyan]NeuroVision AutoML Pipeline[/bold cyan]\n"
            "[dim]Autonomous AI/ML — Zero intervention required[/dim]",
            border_style="cyan"
        ))

        # Step 1: Load data
        console.print("\n[bold yellow]Step 1:[/bold yellow] Loading dataset...")
        df = self._load_data(data_path)
        console.print(f"  [green]Dataset loaded:[/green] {df.shape[0]} rows x {df.shape[1]} columns")

        # Step 2: Anomaly detection
        if detect_anomalies:
            console.print("\n[bold yellow]Step 2:[/bold yellow] Running anomaly detection...")
            anomaly_report = self.anomaly_detector.detect(df.drop(columns=[target_col]))
            n_anomalies = anomaly_report["n_anomalies"]
            console.print(f"  [yellow]Anomalies found:[/yellow] {n_anomalies} rows flagged")
            self.results["anomaly_report"] = anomaly_report

        # Step 3: Preprocess
        console.print("\n[bold yellow]Step 3:[/bold yellow] Smart preprocessing & feature engineering...")
        X, y = self.preprocessor.fit_transform(df, target_col)
        if problem_type is None:
            problem_type = self.preprocessor.detected_problem_type
        console.print(f"  [green]Problem type detected:[/green] [bold]{problem_type.upper()}[/bold]")
        console.print(f"  [green]Features after engineering:[/green] {X.shape[1]} columns")

        # Step 4: Train all models
        console.print("\n[bold yellow]Step 4:[/bold yellow] Training all models in parallel...")
        leaderboard = self.trainer.train_all(
            X, y, problem_type=problem_type, test_size=test_size
        )
        self._print_leaderboard(leaderboard, problem_type)

        # Step 5: Best model
        best_model_name = leaderboard[0]["model_name"]
        best_model = leaderboard[0]["model"]
        console.print(f"\n  [bold green]Best Model:[/bold green] {best_model_name}")

        # Step 6: SHAP Explainability
        if explain:
            console.print("\n[bold yellow]Step 6:[/bold yellow] Generating SHAP explanations...")
            X_test = leaderboard[0].get("X_test")
            shap_values = self.explainer.explain(best_model, X_test, problem_type)
            self.results["shap_values"] = shap_values
            console.print("  [green]SHAP explanation complete.[/green]")

        elapsed = time.time() - start_time
        console.print(f"\n[bold green]Pipeline complete in {elapsed:.2f}s[/bold green]")
        console.print("[bold cyan]Launch dashboard with: streamlit run app.py[/bold cyan]\n")

        self.results.update({
            "leaderboard": leaderboard,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "problem_type": problem_type,
            "feature_names": list(X.columns),
            "elapsed_seconds": elapsed,
        })
        return self.results

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load dataset from CSV or Excel file."""
        path = Path(data_path)
        if path.suffix == ".csv":
            return pd.read_csv(data_path)
        elif path.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(data_path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _print_leaderboard(self, leaderboard: list, problem_type: str) -> None:
        """Print a rich-formatted model leaderboard."""
        table = Table(title="Model Leaderboard", border_style="cyan")
        table.add_column("Rank", style="bold", width=6)
        table.add_column("Model", style="cyan", width=25)

        if problem_type == "classification":
            table.add_column("Accuracy", style="green")
            table.add_column("F1 Score", style="yellow")
            table.add_column("AUC-ROC", style="magenta")
            for i, entry in enumerate(leaderboard, 1):
                table.add_row(
                    str(i),
                    entry["model_name"],
                    f"{entry.get('accuracy', 0):.4f}",
                    f"{entry.get('f1', 0):.4f}",
                    f"{entry.get('roc_auc', 0):.4f}",
                )
        else:
            table.add_column("RMSE", style="green")
            table.add_column("MAE", style="yellow")
            table.add_column("R2 Score", style="magenta")
            for i, entry in enumerate(leaderboard, 1):
                table.add_row(
                    str(i),
                    entry["model_name"],
                    f"{entry.get('rmse', 0):.4f}",
                    f"{entry.get('mae', 0):.4f}",
                    f"{entry.get('r2', 0):.4f}",
                )
        console.print(table)

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """Run prediction using the best trained model."""
        if "best_model" not in self.results:
            raise RuntimeError("Run engine.run() before calling predict()")
        X_processed = self.preprocessor.transform(X_new)
        return self.results["best_model"].predict(X_processed)

    def predict_proba(self, X_new: pd.DataFrame) -> np.ndarray:
        """Run probability prediction (classification only)."""
        if "best_model" not in self.results:
            raise RuntimeError("Run engine.run() before calling predict_proba()")
        X_processed = self.preprocessor.transform(X_new)
        model = self.results["best_model"]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_processed)
        raise ValueError("Best model does not support probability prediction")
