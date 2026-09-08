"""Leakage-safe feature construction for game-level modeling."""
from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

EARLY_FEATURE_COLUMNS = ["home_advantage", "home_win_rate", "away_win_rate", "home_wins", "home_losses", "away_wins", "away_losses", "home_streak", "away_streak", "home_runs_for", "home_runs_against", "away_runs_for", "away_runs_against", "home_rest", "away_rest", "home_pitcher_known", "away_pitcher_known", "home_pitcher_wins", "home_pitcher_losses", "away_pitcher_wins", "away_pitcher_losses", "home_pitcher_era", "away_pitcher_era"]
ENHANCED_FEATURE_COLUMNS = EARLY_FEATURE_COLUMNS + ["home_lineup_strength", "away_lineup_strength", "lineup_complete"]
FEATURE_COLUMNS = EARLY_FEATURE_COLUMNS


def _lineup_count(value) -> float:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return 0.0
    if isinstance(value, (list, tuple)):
        return float(sum(bool(str(player).strip()) for player in value))
    return 0.0


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build simple pre-game features; rows must be chronologically ordered."""
    if games.empty:
        return pd.DataFrame(columns=ENHANCED_FEATURE_COLUMNS + ["game_pk", "home_win"])
    frame = games.sort_values("game_date").copy()
    for col in ("home_score", "away_score"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    frame["home_runs_for"] = frame.groupby("home_team_id")["home_score"].transform(lambda s: s.shift().expanding().mean()).fillna(4.5)
    frame["home_runs_against"] = frame.groupby("home_team_id")["away_score"].transform(lambda s: s.shift().expanding().mean()).fillna(4.5)
    frame["away_runs_for"] = frame.groupby("away_team_id")["away_score"].transform(lambda s: s.shift().expanding().mean()).fillna(4.5)
    frame["away_runs_against"] = frame.groupby("away_team_id")["home_score"].transform(lambda s: s.shift().expanding().mean()).fillna(4.5)
    records = {}
    pitcher_records = {}
    record_features = []
    current_season = None
    for _, row in frame.iterrows():
        season = str(row["game_date"])[:4]
        if season != current_season:
            records, pitcher_records = {}, {}
            current_season = season
        home_state = records.setdefault(row["home_team_id"], {"wins": 0, "losses": 0, "streak": 0})
        away_state = records.setdefault(row["away_team_id"], {"wins": 0, "losses": 0, "streak": 0})
        home_games = home_state["wins"] + home_state["losses"]
        away_games = away_state["wins"] + away_state["losses"]
        home_pitcher = pitcher_records.setdefault(row.get("home_probable_pitcher_id"), {"wins": 0, "losses": 0})
        away_pitcher = pitcher_records.setdefault(row.get("away_probable_pitcher_id"), {"wins": 0, "losses": 0})
        record_features.append({"home_wins": home_state["wins"], "home_losses": home_state["losses"], "away_wins": away_state["wins"], "away_losses": away_state["losses"], "home_win_rate": home_state["wins"] / home_games if home_games else 0.5, "away_win_rate": away_state["wins"] / away_games if away_games else 0.5, "home_streak": home_state["streak"], "away_streak": away_state["streak"], "home_pitcher_wins": home_pitcher["wins"], "home_pitcher_losses": home_pitcher["losses"], "away_pitcher_wins": away_pitcher["wins"], "away_pitcher_losses": away_pitcher["losses"]})
        if row["home_score"] > row["away_score"]:
            home_state["wins"] += 1; away_state["losses"] += 1
            home_state["streak"] = home_state["streak"] + 1 if home_state["streak"] >= 0 else 1
            away_state["streak"] = away_state["streak"] - 1 if away_state["streak"] <= 0 else -1
        elif row["away_score"] > row["home_score"]:
            away_state["wins"] += 1; home_state["losses"] += 1
            away_state["streak"] = away_state["streak"] + 1 if away_state["streak"] >= 0 else 1
            home_state["streak"] = home_state["streak"] - 1 if home_state["streak"] <= 0 else -1
        if row.get("home_probable_pitcher_id") is not None:
            if row["home_score"] > row["away_score"]: home_pitcher["wins"] += 1
            else: home_pitcher["losses"] += 1
        if row.get("away_probable_pitcher_id") is not None:
            if row["away_score"] > row["home_score"]: away_pitcher["wins"] += 1
            else: away_pitcher["losses"] += 1
    record_frame = pd.DataFrame(record_features, index=frame.index)
    for column in ("home_wins", "home_losses", "away_wins", "away_losses", "home_win_rate", "away_win_rate", "home_streak", "away_streak", "home_pitcher_wins", "home_pitcher_losses", "away_pitcher_wins", "away_pitcher_losses"):
        frame[column] = record_frame[column]
    dates = pd.to_datetime(frame["game_date"])
    frame["home_rest"] = dates.groupby(frame["home_team_id"]).diff().dt.days.fillna(3).clip(lower=0, upper=14)
    frame["away_rest"] = dates.groupby(frame["away_team_id"]).diff().dt.days.fillna(3).clip(lower=0, upper=14)
    frame["home_advantage"] = 1.0
    frame["home_pitcher_known"] = frame["home_probable_pitcher_id"].notna().astype(float)
    frame["away_pitcher_known"] = frame["away_probable_pitcher_id"].notna().astype(float)
    frame["home_pitcher_era"] = pd.to_numeric(frame["home_pitcher_era"] if "home_pitcher_era" in frame else pd.Series(4.20, index=frame.index), errors="coerce").fillna(4.20)
    frame["away_pitcher_era"] = pd.to_numeric(frame["away_pitcher_era"] if "away_pitcher_era" in frame else pd.Series(4.20, index=frame.index), errors="coerce").fillna(4.20)
    for side in ("home", "away"):
        strength = pd.to_numeric(frame[f"{side}_lineup_strength"], errors="coerce") if f"{side}_lineup_strength" in frame else pd.Series(float("nan"), index=frame.index)
        lineup = frame[f"{side}_lineup"] if f"{side}_lineup" in frame else frame.get(f"{side}_lineup_json", pd.Series(index=frame.index, dtype=object))
        frame[f"{side}_lineup_strength"] = strength.fillna(lineup.apply(_lineup_count)).fillna(0.0)
    frame["lineup_complete"] = ((frame["home_lineup_strength"] >= 9) & (frame["away_lineup_strength"] >= 9)).astype(float)
    frame["home_win"] = (frame["home_score"] > frame["away_score"]).astype(int)
    return frame[["game_pk", "home_win"] + ENHANCED_FEATURE_COLUMNS]


def build_prediction_features(game: Mapping, historical_games: pd.DataFrame) -> dict[str, float]:
    """Build live features with the same definitions used for training."""
    candidate = dict(game)
    if "game_pk" not in candidate:
        raise ValueError("A prediction game must include game_pk.")
    history = historical_games.copy() if historical_games is not None else pd.DataFrame()
    combined = pd.concat([history, pd.DataFrame([candidate])], ignore_index=True, sort=False)
    built = build_features(combined)
    match = built[built["game_pk"] == candidate["game_pk"]]
    if len(match) != 1:
        raise ValueError("Prediction game must have a unique game_pk.")
    row = match.iloc[0]
    return {column: float(row[column]) for column in ENHANCED_FEATURE_COLUMNS}
