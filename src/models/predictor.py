"""Prediction service with a trained-model and baseline fallback."""
from pathlib import Path

import pandas as pd

from .baseline import predict_game
from .trainer import load_model


def _fallback_prediction(home_team: str, away_team: str, features: dict):
    return predict_game(
        home_team,
        away_team,
        home_pitcher=features.get("home_pitcher", ""),
        away_pitcher=features.get("away_pitcher", ""),
        home_lineup=features.get("home_lineup", ()),
        away_lineup=features.get("away_lineup", ()),
    )


def predict_with_model(home_team: str, away_team: str, features: dict, early_path: str = "models/mlb_early_model.joblib", enhanced_path: str = "models/mlb_enhanced_model.joblib"):
    complete = features.get("lineup_complete", 0) >= 1
    artifact_path = enhanced_path if complete and Path(enhanced_path).exists() else early_path
    if not Path(artifact_path).exists():
        return _fallback_prediction(home_team, away_team, features), "deterministic-baseline"
    artifact = load_model(artifact_path)
    if not artifact.get("version", "").startswith(("early-logistic-calibrated-v2", "enhanced-logistic-calibrated-v2")):
        return _fallback_prediction(home_team, away_team, features), "outdated-model-retrain-required"
    model_frame = pd.DataFrame([features]).reindex(columns=artifact["features"], fill_value=0.0)
    probability = float(artifact["model"].predict_proba(model_frame)[0, 1])
    from src.models.baseline import GamePrediction
    mode = "enhanced" if artifact_path == enhanced_path else "early"
    result = GamePrediction(probability, 1.0 - probability, f"{mode.title()} calibrated model using the available pre-game data.")
    return result, artifact["version"]
