"""
Soccer AI Betting Predictor – Core package.
Exposes main components for external scripts and tests.
"""

from .collector import DataCollector
from .features import FeatureEngineer
from .model import EnsemblePredictor
from .value_finder import ValueFinder
from .feedback import FeedbackLoop
from .pipeline import Pipeline

__all__ = [
    "DataCollector",
    "FeatureEngineer",
    "EnsemblePredictor",
    "ValueFinder",
    "FeedbackLoop",
    "Pipeline",
]
