"""Detailed reports for probability forecasts."""
from __future__ import annotations

import pandas as pd

from .metrics import Evaluation, evaluate_probabilities


def calibration_table(actual: pd.Series, probabilities, bins: int = 10) -> pd.DataFrame:
    """Summarize predicted probability versus observed win rate by bin."""
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    actual = pd.Series(actual).astype(int).reset_index(drop=True)
    probabilities = pd.Series(probabilities, dtype=float).clip(0.0, 1.0).reset_index(drop=True)
    if len(actual) != len(probabilities) or actual.empty:
        raise ValueError("actual and probabilities must have the same non-zero length.")
    result = pd.DataFrame({"actual_home_win": actual, "home_probability": probabilities})
    result["probability_bin"] = pd.cut(result["home_probability"], bins=bins, labels=False, include_lowest=True)
    summary = result.groupby("probability_bin", observed=False).agg(
        n_samples=("actual_home_win", "size"),
        mean_predicted=("home_probability", "mean"),
        observed_rate=("actual_home_win", "mean"),
    ).reset_index()
    summary["calibration_gap"] = (summary["mean_predicted"] - summary["observed_rate"]).abs()
    return summary


def metrics_by_season(predictions: pd.DataFrame) -> dict[str, Evaluation]:
    """Calculate the same probability metrics separately for each season."""
    required = {"actual_home_win", "home_probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = predictions.copy()
    if "season" not in frame:
        if "game_date" in frame:
            dates = frame["game_date"]
        elif "game_datetime" in frame:
            dates = frame["game_datetime"]
        else:
            raise ValueError("Predictions require game_date, game_datetime, or season.")
        frame["season"] = pd.to_datetime(dates, errors="coerce", utc=True).dt.year
    frame = frame.dropna(subset=["season"])
    return {
        str(season): evaluate_probabilities(group["actual_home_win"], group["home_probability"])
        for season, group in frame.groupby("season", sort=True)
    }
