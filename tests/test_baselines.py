import pandas as pd
import pytest

from src.evaluation.baselines import elo_baseline, evaluate_baselines, home_team_baseline


def _frame():
    return pd.DataFrame(
        {
            "game_pk": [3, 1, 2, 4],
            "game_datetime": ["2026-04-03", "2026-04-01", "2026-04-02", "2026-04-04"],
            "home_team_id": [1, 1, 2, 2],
            "away_team_id": [2, 2, 1, 1],
            "home_win": [1, 0, 1, 0],
        }
    )


def test_home_baseline_returns_probability_metrics():
    result = home_team_baseline(_frame())

    assert result.name == "home-team-baseline"
    assert result.metrics.n_samples == 4
    assert result.metrics.log_loss == pytest.approx(0.693147, rel=1e-5)


def test_elo_baseline_is_chronological_and_updates_after_games():
    result = elo_baseline(_frame())
    sorted_result = elo_baseline(_frame().sort_values("game_datetime"))

    assert result.name == "elo-baseline"
    assert result.metrics.n_samples == 4
    assert 0 <= result.metrics.brier_score <= 1
    assert result.metrics.to_dict() == sorted_result.metrics.to_dict()


def test_evaluate_baselines_returns_both_benchmarks():
    results = evaluate_baselines(_frame())

    assert set(results) == {"home_team", "elo"}
