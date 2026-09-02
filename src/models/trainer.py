"""Training and evaluation for the first calibrated model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from .features import EARLY_FEATURE_COLUMNS, ENHANCED_FEATURE_COLUMNS


@dataclass
class Evaluation:
    log_loss: float
    brier_score: float
    accuracy: float
    roc_auc: float


def chronological_folds(n_samples: int):
    """Return time-ordered calibration folds with no future training rows."""
    if n_samples < 8:
        raise ValueError("At least 8 rows are required for chronological folds.")
    return list(TimeSeriesSplit(n_splits=3).split(range(n_samples)))


def _fit(frame: pd.DataFrame, features: list[str], artifact_path: str, version: str) -> Evaluation:
    frame = frame.sort_values("game_pk").dropna(subset=features + ["home_win"])
    if len(frame) < 30:
        raise ValueError("At least 30 historical games are required to train the model.")
    split = max(1, int(len(frame) * 0.8))
    train, test = frame.iloc[:split], frame.iloc[split:]
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    # Calibration folds must remain chronological; ordinary K-fold would train
    # on games that occur after the validation game.
    model = CalibratedClassifierCV(base, method="sigmoid", cv=chronological_folds(len(train)))
    model.fit(train[features], train["home_win"])
    probabilities = model.predict_proba(test[features])[:, 1]
    metrics = Evaluation(log_loss(test["home_win"], probabilities), brier_score_loss(test["home_win"], probabilities), accuracy_score(test["home_win"], probabilities >= 0.5), roc_auc_score(test["home_win"], probabilities))
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "version": version}, artifact_path)
    return metrics


def train_models(frame: pd.DataFrame, early_path: str = "models/mlb_early_model.joblib", enhanced_path: str = "models/mlb_enhanced_model.joblib") -> dict[str, Evaluation]:
    """Train the early model always and enhanced model when lineups are sufficient."""
    results = {"early": _fit(frame, EARLY_FEATURE_COLUMNS, early_path, "early-logistic-calibrated-v2-record-streak")}
    complete = frame[frame["lineup_complete"] >= 1] if "lineup_complete" in frame else frame.iloc[0:0]
    if len(complete) >= 30:
        results["enhanced"] = _fit(complete, ENHANCED_FEATURE_COLUMNS, enhanced_path, "enhanced-logistic-calibrated-v2-record-streak")
    return results


def train_model(frame: pd.DataFrame, artifact_path: str = "models/mlb_early_model.joblib") -> Evaluation:
    """Backward-compatible alias for training the early model."""
    return _fit(frame, EARLY_FEATURE_COLUMNS, artifact_path, "early-logistic-calibrated-v2-record-streak")


def load_model(artifact_path: str = "models/mlb_win_model.joblib"):
    return joblib.load(artifact_path)
