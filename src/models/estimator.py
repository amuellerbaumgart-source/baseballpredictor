"""Shared estimator construction for training and time-ordered evaluation."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chronological_folds(n_samples: int):
    """Return calibration folds where validation rows never precede training rows."""
    if n_samples < 8:
        raise ValueError("At least 8 rows are required for chronological folds.")
    return list(TimeSeriesSplit(n_splits=3).split(range(n_samples)))


def build_calibrated_estimator(n_samples: int):
    """Build the calibrated estimator used by both training and evaluation."""
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    return CalibratedClassifierCV(base, method="sigmoid", cv=chronological_folds(n_samples))
