"""
NeuroVision-AutoML
An autonomous AI/ML pipeline that auto-selects, trains, evaluates,
and explains the best ML model for any dataset.

Author: Surya Ethan
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Surya Ethan"

from neurovision.engine import NeuroVisionEngine
from neurovision.preprocessor import SmartPreprocessor
from neurovision.trainer import ModelTrainer
from neurovision.explainer import SHAPExplainer
from neurovision.anomaly import AnomalyDetector

__all__ = [
    "NeuroVisionEngine",
    "SmartPreprocessor",
    "ModelTrainer",
    "SHAPExplainer",
    "AnomalyDetector",
]
