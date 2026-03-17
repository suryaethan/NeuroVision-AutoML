"""
NeuroVision AutoML - CLI Entry Point
Run the full AutoML pipeline from the command line.

Usage:
    python main.py --data path/to/data.csv --target target_column
    python main.py --data data.csv --target price --problem regression
    python main.py --data data.csv --target label --no-anomalies --no-shap
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from neurovision.engine import NeuroVisionEngine

app = typer.Typer(
    name="neurovision",
    help="🧠 NeuroVision AutoML - Autonomous ML Pipeline",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV dataset"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
    problem: Optional[str] = typer.Option(
        None, "--problem", "-p",
        help="Problem type: 'classification' or 'regression' (auto-detect if not set)"
    ),
    test_size: float = typer.Option(0.2, "--test-size", help="Test split ratio"),
    anomalies: bool = typer.Option(True, "--anomalies/--no-anomalies", help="Run anomaly detection"),
    shap: bool = typer.Option(True, "--shap/--no-shap", help="Generate SHAP explanations"),
    random_state: int = typer.Option(42, "--seed", help="Random seed"),
):
    """
    Run the full NeuroVision AutoML pipeline on a dataset.
    """
    # Validate file
    if not Path(data).exists():
        console.print(f"[bold red]ERROR:[/bold red] File not found: {data}")
        raise typer.Exit(code=1)

    if problem and problem not in ["classification", "regression"]:
        console.print("[bold red]ERROR:[/bold red] --problem must be 'classification' or 'regression'")
        raise typer.Exit(code=1)

    console.print(Panel.fit(
        "[bold cyan]🧠 NeuroVision AutoML[/bold cyan]\n"
        f"[dim]Dataset: {data} | Target: {target}[/dim]",
        border_style="cyan"
    ))

    # Run pipeline
    engine = NeuroVisionEngine(random_state=random_state)
    results = engine.run(
        data_path=data,
        target_col=target,
        problem_type=problem,
        test_size=test_size,
        detect_anomalies=anomalies,
        explain=shap,
    )

    # Summary
    console.print("\n[bold green]Summary[/bold green]")
    console.print(f"  Best Model  : [cyan]{results['best_model_name']}[/cyan]")
    console.print(f"  Problem Type: [cyan]{results['problem_type']}[/cyan]")
    console.print(f"  Time Taken  : [cyan]{results['elapsed_seconds']:.2f}s[/cyan]")
    console.print(f"  Features    : [cyan]{len(results['feature_names'])}[/cyan]")
    console.print("\n[bold cyan]To view dashboard:[/bold cyan] streamlit run app.py")


if __name__ == "__main__":
    app()
