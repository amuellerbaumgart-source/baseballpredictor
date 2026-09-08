import pandas as pd
import pytest
from src.models.features import ENHANCED_FEATURE_COLUMNS, FEATURE_COLUMNS, build_features, build_prediction_features
from src.models.trainer import chronological_folds

def test_features_do_not_use_current_game_score():
    games = pd.DataFrame([{"game_pk": 1, "game_date": "2026-04-01", "home_team_id": 1, "away_team_id": 2, "home_score": 10, "away_score": 0, "home_probable_pitcher_id": None, "away_probable_pitcher_id": None}, {"game_pk": 2, "game_date": "2026-04-02", "home_team_id": 1, "away_team_id": 2, "home_score": 2, "away_score": 1, "home_probable_pitcher_id": 3, "away_probable_pitcher_id": 4}])
    features = build_features(games)
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features.loc[features.game_pk == 1, "home_runs_for"].iloc[0] == 4.5


def test_future_game_cannot_change_prior_features():
    games = pd.DataFrame([{"game_pk": 1, "game_date": "2026-04-01", "home_team_id": 1, "away_team_id": 2, "home_score": 3, "away_score": 2, "home_probable_pitcher_id": 10, "away_probable_pitcher_id": 20}, {"game_pk": 2, "game_date": "2026-04-02", "home_team_id": 1, "away_team_id": 2, "home_score": 4, "away_score": 1, "home_probable_pitcher_id": 10, "away_probable_pitcher_id": 20}])
    changed = games.copy()
    changed.loc[1, "home_score"], changed.loc[1, "away_score"] = 0, 20
    first, changed_first = build_features(games).iloc[0], build_features(changed).iloc[0]
    assert first["home_wins"] == changed_first["home_wins"]
    assert first["home_runs_for"] == changed_first["home_runs_for"]


def test_calibration_folds_are_time_ordered():
    for train_indices, validation_indices in chronological_folds(30):
        assert max(train_indices) < min(validation_indices)


def test_training_and_prediction_use_the_same_feature_schema_and_values():
    history = pd.DataFrame([{
        "game_pk": 1, "game_date": "2026-04-01", "home_team_id": 1, "away_team_id": 2,
        "home_score": 5, "away_score": 3, "home_probable_pitcher_id": 10,
        "away_probable_pitcher_id": 20, "home_lineup_json": '["A"]', "away_lineup_json": '[]',
    }])
    candidate = {
        "game_pk": 2, "game_date": "2026-04-02", "home_team_id": 1, "away_team_id": 2,
        "home_probable_pitcher_id": 10, "away_probable_pitcher_id": 20,
        "home_pitcher_era": 3.5, "away_pitcher_era": 4.5,
        "home_lineup": ["A", "B"], "away_lineup": ["C"],
    }

    training_frame = build_features(pd.concat([history, pd.DataFrame([candidate])], ignore_index=True, sort=False))
    prediction_features = build_prediction_features(candidate, history)

    assert list(prediction_features) == ENHANCED_FEATURE_COLUMNS
    assert list(training_frame.columns) == ["game_pk", "home_win"] + ENHANCED_FEATURE_COLUMNS
    expected = training_frame.loc[training_frame["game_pk"] == 2, ENHANCED_FEATURE_COLUMNS].iloc[0].to_dict()
    assert prediction_features == pytest.approx(expected)
