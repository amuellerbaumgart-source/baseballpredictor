from datetime import date

import pandas as pd

from src.data.store import Store
from src.models.features import build_features


def _game(game_pk, game_datetime, home_score=0, away_score=0, **extra):
    return {"game_pk": game_pk, "game_date": game_datetime[:10], "game_datetime": game_datetime, "home_team_id": 1, "away_team_id": 2, "home_score": home_score, "away_score": away_score, "home_probable_pitcher_id": 10, "away_probable_pitcher_id": 20, **extra}


def test_same_day_games_are_ordered_by_datetime_then_game_pk():
    games = pd.DataFrame([_game(2, "2026-04-01T20:00:00Z"), _game(3, "2026-04-01T19:00:00Z"), _game(1, "2026-04-01T19:00:00Z", home_score=5, away_score=1)])

    features = build_features(games)

    assert list(features["game_pk"]) == [1, 3, 2]
    assert features.loc[features["game_pk"] == 3, "home_wins"].iloc[0] == 1


def test_pitcher_era_after_game_time_is_not_used():
    games = pd.DataFrame([_game(1, "2026-04-01T19:00:00Z", home_pitcher_era=2.1, home_pitcher_era_as_of="2026-04-02T00:00:00Z")])

    features = build_features(games)

    assert features.loc[0, "home_pitcher_era"] == 4.20


def test_future_pitcher_snapshot_cannot_change_earlier_features(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    store.upsert_pitcher_stats({"pitcher_id": 10, "season": 2026, "era": 2.1, "wins": 1, "losses": 0, "as_of_datetime": "2026-04-02T00:00:00Z"})

    assert store.pitcher_stats(10, 2026, "2026-04-01T23:00:00Z")["available"] is False
    assert store.pitcher_stats(10, 2026, "2026-04-02T00:00:00Z")["era"] == 2.1


def test_completed_games_do_not_join_unqualified_season_pitcher_stats(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    store.upsert_games([{"gamePk": 1, "officialDate": "2026-04-01", "gameDate": "2026-04-01T19:00:00Z", "status": {"abstractGameState": "Final"}, "gameType": "R", "teams": {"home": {"team": {"id": 1}, "score": 5, "probablePitcher": {"id": 10}}, "away": {"team": {"id": 2}, "score": 3, "probablePitcher": {"id": 20}}}}])
    store.upsert_pitcher_stats({"pitcher_id": 10, "season": 2026, "era": 2.1, "as_of_datetime": "2026-10-01T00:00:00Z"})

    row = store.completed_games()[0]

    assert "home_pitcher_era" not in row.keys()
