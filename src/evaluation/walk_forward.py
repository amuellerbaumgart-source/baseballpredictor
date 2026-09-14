"""Leakage-safe expanding-window evaluation for game forecasts."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import Evaluation, evaluate_probabilities


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample predictions and aggregate probability metrics."""

    predictions: pd.DataFrame
    metrics: Evaluation


def evaluate_walk_forward(
    frame: pd.DataFrame,
    features: list[str],
    min_train_size: int = 30,
) -> WalkForwardResult:
    """Score each game using only earlier games in an expanding training window."""
    if min_train_size < 2:
        raise ValueError("min_train_size must be at least 2.")
    missing = [column for column in [*features, "home_win"] if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    sort_columns = [column for column in ("game_datetime", "game_date", "game_pk") if column in frame]
    eligible = frame.sort_values(sort_columns, kind="mergesort").dropna(subset=[*features, "home_win"]).reset_index(drop=True)
    if len(eligible) <= min_train_size:
        raise ValueError("At least one game must remain after the minimum training window.")

    predictions: list[dict[str, object]] = []
    for test_index in range(min_train_size, len(eligible)):
        train = eligible.iloc[:test_index]
        if train["home_win"].nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        model.fit(train[features], train["home_win"].astype(int))
        probability = float(model.predict_proba(eligible.iloc[[test_index]][features])[0, 1])
        row = eligible.iloc[test_index]
        predictions.append(
            {
                "game_pk": row.get("game_pk"),
                "game_datetime": row.get("game_datetime"),
                "season": pd.to_datetime(row.get("game_datetime"), errors="coerce", utc=True).year
                if pd.notna(row.get("game_datetime"))
                else row.get("game_date", "")[:4],
                "actual_home_win": int(row["home_win"]),
                "home_probability": probability,
                "train_rows": test_index,
            }
        )

    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        raise ValueError("No walk-forward predictions could be made with both outcome classes available.")
    metrics = evaluate_probabilities(prediction_frame["actual_home_win"], prediction_frame["home_probability"])
    return WalkForwardResult(prediction_frame, metrics)
