import joblib
import pytest

from src.models.predictor import predict_with_model


FEATURES = {"lineup_complete": 0}


def test_missing_model_uses_deterministic_fallback_for_display_names(tmp_path):
    result, version = predict_with_model("Los Angeles Dodgers", "New York Yankees", FEATURES, early_path=str(tmp_path / "missing.joblib"))

    assert version == "deterministic-baseline"
    assert result.home_win_probability + result.away_win_probability == pytest.approx(1.0)


def test_outdated_model_uses_fallback(tmp_path):
    artifact = tmp_path / "outdated.joblib"
    joblib.dump({"version": "old-model"}, artifact)

    result, version = predict_with_model("LAD", "NYY", FEATURES, early_path=str(artifact))

    assert version == "outdated-model-retrain-required"
    assert result.home_win_probability + result.away_win_probability == pytest.approx(1.0)


@pytest.mark.parametrize(("home", "away"), [("LAD", "NYY"), ("Los Angeles Dodgers", "New York Yankees")])
def test_valid_team_identifiers_are_supported(home, away, tmp_path):
    result, _ = predict_with_model(home, away, FEATURES, early_path=str(tmp_path / "missing.joblib"))

    assert 0 < result.home_win_probability < 1
    assert 0 < result.away_win_probability < 1


def test_same_team_is_rejected_after_name_normalization(tmp_path):
    with pytest.raises(ValueError, match="different"):
        predict_with_model("LAD", "Los Angeles Dodgers", FEATURES, early_path=str(tmp_path / "missing.joblib"))


def test_fallback_uses_available_lineup_inputs(tmp_path):
    result, _ = predict_with_model("LAD", "NYY", {"lineup_complete": 0, "home_lineup": ["A"] * 9}, early_path=str(tmp_path / "missing.joblib"))
    neutral, _ = predict_with_model("LAD", "NYY", FEATURES, early_path=str(tmp_path / "missing.joblib"))

    assert result.home_win_probability > neutral.home_win_probability
