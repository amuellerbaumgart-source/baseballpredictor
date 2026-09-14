"""Simple chronological baselines for judging the game model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .metrics import Evaluation, evaluate_probabilities


@dataclass(frozen=True)
class BaselineResult:
    """Name and metrics for one benchmark forecast."""

    name: str
    metrics: Evaluation


def _eligible_games(frame: pd.DataFrame, require_teams: bool = True) -> pd.DataFrame:
    required = {"home_win"}
    if require_teams:
        required |= {"home_team_id", "away_team_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    sort_columns = [column for column in ("game_datetime", "game_date", "game_pk") if column in frame]
    return frame.sort_values(sort_columns, kind="mergesort").dropna(subset=list(required)).reset_index(drop=True)


def home_team_baseline(frame: pd.DataFrame, probability: float = 0.5, test_start: int = 0) -> BaselineResult:
    """Evaluate a constant home-win probability across the supplied games."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")
    if test_start < 0:
        raise ValueError("test_start cannot be negative.")
    games = _eligible_games(frame, require_teams=False).iloc[test_start:]
    probabilities = [probability] * len(games)
    return BaselineResult("home-team-baseline", evaluate_probabilities(games["home_win"], probabilities))


def elo_baseline(
    frame: pd.DataFrame,
    initial_rating: float = 1500.0,
    home_advantage: float = 50.0,
    k_factor: float = 20.0,
    test_start: int = 0,
) -> BaselineResult:
    """Evaluate an expanding Elo forecast, updating ratings after each game."""
    if k_factor <= 0:
        raise ValueError("k_factor must be positive.")
    if test_start < 0:
        raise ValueError("test_start cannot be negative.")
    games = _eligible_games(frame)
    ratings: dict[object, float] = {}
    actual: list[int] = []
    probabilities: list[float] = []
    for game_index, (_, game) in enumerate(games.iterrows()):
        home, away = game["home_team_id"], game["away_team_id"]
        home_rating = ratings.get(home, initial_rating)
        away_rating = ratings.get(away, initial_rating)
        probability = 1.0 / (1.0 + math.pow(10.0, -(home_rating + home_advantage - away_rating) / 400.0))
        outcome = int(game["home_win"])
        if game_index >= test_start:
            actual.append(outcome)
            probabilities.append(probability)
        else:
            # Keep rating updates for the historical warm-up, but do not score it.
            pass
        change = k_factor * (outcome - probability)
        ratings[home] = home_rating + change
        ratings[away] = away_rating - change
    return BaselineResult("elo-baseline", evaluate_probabilities(actual, probabilities))


def evaluate_baselines(frame: pd.DataFrame, test_start: int = 0) -> dict[str, BaselineResult]:
    """Return the benchmarks used to contextualize the trained model."""
    return {"home_team": home_team_baseline(frame, test_start=test_start), "elo": elo_baseline(frame, test_start=test_start)}
