import pytest
from src.models.baseline import predict_game

def test_probabilities_are_complements():
    result = predict_game("LAD", "NYY")
    assert 0 < result.home_win_probability < 1
    assert result.home_win_probability + result.away_win_probability == pytest.approx(1.0)

def test_complete_home_lineup_has_small_advantage():
    result = predict_game("LAD", "NYY", home_lineup=["x"] * 9, away_lineup=[])
    neutral = predict_game("LAD", "NYY")
    assert result.home_win_probability > neutral.home_win_probability

def test_same_team_is_rejected():
    with pytest.raises(ValueError, match="different"):
        predict_game("LAD", "LAD")
