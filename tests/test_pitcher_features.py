from src.data.mlb_api import MLBClient
from src.data.store import Store


def test_innings_notation_is_converted_to_outs():
    assert MLBClient.innings_to_outs("6.0") == 18
    assert MLBClient.innings_to_outs("6.1") == 19
    assert MLBClient.innings_to_outs("6.2") == 20


def test_game_feed_extracts_pitcher_game_stats():
    feed = {
        "gameData": {"probablePitchers": {}},
        "liveData": {"boxscore": {"teams": {"home": {"players": {"ID10": {"person": {"id": 10}, "stats": {"pitching": {"inningsPitched": "6.2", "earnedRuns": 2}}}}}, "away": {"players": {}}}}},
    }

    details = MLBClient.extract_game_details(feed)

    assert details["pitcher_game_stats"] == [{"pitcher_id": 10, "innings_pitched": 20, "earned_runs": 2}]


def test_historical_era_uses_only_prior_pitcher_appearances(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    store.upsert_games([
        {"gamePk": 1, "officialDate": "2026-04-01", "gameDate": "2026-04-01T19:00:00Z", "status": {"abstractGameState": "Final"}, "gameType": "R", "teams": {"home": {"team": {"id": 1}, "score": 5, "probablePitcher": {"id": 10}}, "away": {"team": {"id": 2}, "score": 3, "probablePitcher": {"id": 20}}}},
        {"gamePk": 2, "officialDate": "2026-04-02", "gameDate": "2026-04-02T19:00:00Z", "status": {"abstractGameState": "Final"}, "gameType": "R", "teams": {"home": {"team": {"id": 1}, "score": 4, "probablePitcher": {"id": 10}}, "away": {"team": {"id": 2}, "score": 2, "probablePitcher": {"id": 20}}}},
    ])
    store.upsert_game_details(1, {"home_lineup": [], "away_lineup": [], "pitcher_game_stats": [{"pitcher_id": 10, "innings_pitched": 18, "earned_runs": 2}]})
    store.upsert_game_details(2, {"home_lineup": [], "away_lineup": [], "pitcher_game_stats": [{"pitcher_id": 10, "innings_pitched": 21, "earned_runs": 0}]})

    rows = store.completed_games()

    assert rows[0]["home_pitcher_era"] is None
    assert rows[1]["home_pitcher_era"] == 1.0
    assert rows[1]["home_pitcher_era_as_of"] == "2026-04-01T19:00:00Z"
