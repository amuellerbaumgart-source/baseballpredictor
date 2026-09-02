from datetime import date
from src.data.mlb_api import MLBClient
from src.data.store import Store

class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload

class FakeSession:
    def get(self, url, **kwargs):
        assert url.endswith("/schedule")
        return FakeResponse({"dates": [{"games": [{"gamePk": 1}]}]})

def test_schedule_parses_games():
    assert MLBClient(session=FakeSession()).games(date(2026, 4, 1)) == [{"gamePk": 1}]


def test_normalize_game_creates_readable_local_summary():
    game = {"gamePk": 9, "officialDate": "2026-09-02", "gameDate": "2026-09-02T22:40:00Z", "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"}, "teams": {"home": {"team": {"id": 113, "name": "Cincinnati Reds", "abbreviation": "CIN"}}, "away": {"team": {"id": 114, "name": "San Diego Padres", "abbreviation": "SD"}}}, "venue": {"name": "Great American Ball Park", "location": {"city": "Cincinnati"}, "timeZone": "America/New_York"}}
    normalized = MLBClient.normalize_game(game)
    assert normalized["away_team_name"] == "San Diego Padres"
    assert normalized["venue_name"] == "Great American Ball Park"
    assert normalized["scheduled_local_time"].startswith("2026-09-02T18:40:00")

def test_store_upserts_and_reads_games(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    store.upsert_games([{"gamePk": 1, "officialDate": "2026-04-01", "gameDate": "2026-04-01T19:00:00Z", "status": {"abstractGameState": "Final"}, "teams": {"home": {"team": {"id": 10}, "score": 5}, "away": {"team": {"id": 20}, "score": 3}}}])
    assert len(store.completed_games()) == 1


def test_team_record_ignores_final_rows_without_scores(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    store.upsert_games([{"gamePk": 1, "officialDate": "2026-04-01", "status": {"abstractGameState": "Final"}, "teams": {"home": {"team": {"id": 10}}, "away": {"team": {"id": 20}}}}])
    assert store.team_record(10) == {"wins": 0, "losses": 0, "streak": 0}


def test_records_are_limited_to_requested_season_and_pitcher_record(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    rows = []
    for pk, year, home_score, away_score in [(1, "2025", 5, 2), (2, "2026", 1, 4), (3, "2026", 3, 1)]:
        rows.append({"gamePk": pk, "officialDate": f"{year}-04-01", "status": {"abstractGameState": "Final"}, "teams": {"home": {"team": {"id": 10}, "score": home_score, "probablePitcher": {"id": 100}}, "away": {"team": {"id": 20}, "score": away_score, "probablePitcher": {"id": 200}}}})
    store.upsert_games(rows)
    assert store.team_record(10, 2026) == {"wins": 1, "losses": 1, "streak": 1}
    assert store.pitcher_record(100, 2026)["wins"] == 1
    assert store.pitcher_record(100, 2026)["losses"] == 1


def test_missing_pitcher_stats_use_unavailable_fallback(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    assert store.pitcher_stats(999999, 2026) == {"era": None, "wins": 0, "losses": 0, "available": False}


def test_repeated_game_refresh_is_idempotent_and_preserves_details(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    game = {"gamePk": 7, "officialDate": "2026-04-01", "gameDate": "2026-04-01T19:00:00Z", "status": {"abstractGameState": "Scheduled"}, "teams": {"home": {"team": {"id": 10, "abbreviation": "CIN"}}, "away": {"team": {"id": 20, "abbreviation": "SD"}}}}
    store.upsert_games([game])
    store.upsert_game_details(7, {"home_pitcher_name": "Home Starter", "away_pitcher_name": "Away Starter", "home_lineup": ["A"], "away_lineup": ["B"]})
    store.upsert_games([game])
    row = store.upcoming_games()[0]
    assert store.count_games() == 1
    assert row["home_pitcher_name"] == "Home Starter"
    assert row["home_lineup_json"] == '["A"]'
