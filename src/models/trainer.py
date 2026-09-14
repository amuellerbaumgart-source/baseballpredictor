"""Training and evaluation for the first calibrated model."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.metrics import Evaluation, evaluate_probabilities

from .estimator import build_calibrated_estimator
from .estimator import chronological_folds as _chronological_folds
from .features import EARLY_FEATURE_COLUMNS, ENHANCED_FEATURE_COLUMNS


def chronological_folds(n_samples: int):
    """Backward-compatible export for the chronological calibration folds."""
    return _chronological_folds(n_samples)


def _fit(frame: pd.DataFrame, features: list[str], artifact_path: str, version: str) -> Evaluation:
    sort_columns = [column for column in ("game_datetime", "game_date", "game_pk") if column in frame]
    frame = frame.sort_values(sort_columns, kind="mergesort").dropna(subset=features + ["home_win"])
    if len(frame) < 30:
        raise ValueError("At least 30 historical games are required to train the model.")
    split = max(1, int(len(frame) * 0.8))
    train, test = frame.iloc[:split], frame.iloc[split:]
    model = build_calibrated_estimator(len(train))
    model.fit(train[features], train["home_win"])
    probabilities = model.predict_proba(test[features])[:, 1]
    metrics = evaluate_probabilities(test["home_win"], probabilities)
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "training_start": str(train.iloc[0].get("game_datetime", train.iloc[0].get("game_date", ""))),
        "training_end": str(train.iloc[-1].get("game_datetime", train.iloc[-1].get("game_date", ""))),
    }
    artifact = {"model": model, "features": features, "version": version, "evaluation": metrics.to_dict(), "training_rows": len(train), "test_rows": len(test), "metadata": metadata}
    existing_metrics = None
    if Path(artifact_path).exists():
        try:
            existing_metrics = joblib.load(artifact_path).get("evaluation", {}).get("log_loss")
        except (OSError, ValueError, KeyError, AttributeError):
            existing_metrics = None
    if existing_metrics is None or metrics.log_loss <= float(existing_metrics):
        joblib.dump(artifact, artifact_path)
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
