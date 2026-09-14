import json

import pytest

from src.data.store import Store


def test_forecast_snapshot_and_prediction_are_persisted(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))
    store.upsert_games([{
        "gamePk": 1,
        "officialDate": "2026-04-01",
        "gameDate": "2026-04-01T19:00:00Z",
        "status": {"abstractGameState": "Scheduled"},
        "teams": {"home": {"team": {"id": 10}}, "away": {"team": {"id": 20}}},
    }])

    store.save_feature_snapshot(1, {"home_rest": 2.0}, "baseline")
    store.save_prediction(1, 0.6, 0.4, "baseline", "test")

    with store.connect() as db:
        snapshot = db.execute("SELECT * FROM feature_snapshots WHERE game_pk=1").fetchone()
        prediction = db.execute("SELECT * FROM predictions WHERE game_pk=1").fetchone()

    assert json.loads(snapshot["feature_json"])["home_rest"] == 2.0
    assert prediction["home_probability"] == 0.6


def test_prediction_rejects_invalid_probabilities(tmp_path):
    store = Store(str(tmp_path / "mlb.sqlite"))

    with pytest.raises(ValueError, match="sum to 1"):
        store.save_prediction(1, 0.8, 0.8, "baseline", "test")
