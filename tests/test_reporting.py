import pandas as pd
import pytest

from src.evaluation.reporting import calibration_table, metrics_by_season


def _predictions():
    return pd.DataFrame(
        {
            "game_date": ["2025-04-01", "2025-04-02", "2026-04-01", "2026-04-02"],
            "actual_home_win": [0, 1, 1, 1],
            "home_probability": [0.2, 0.8, 0.7, 0.9],
        }
    )


def test_calibration_table_reports_observed_rate_and_gap():
    table = calibration_table(_predictions()["actual_home_win"], _predictions()["home_probability"], bins=2)

    assert list(table.columns) == ["probability_bin", "n_samples", "mean_predicted", "observed_rate", "calibration_gap"]
    assert table["n_samples"].sum() == 4
    assert (table["calibration_gap"] >= 0).all()


def test_calibration_table_uses_fixed_probability_range():
    table = calibration_table(pd.Series([0, 1]), [0.49, 0.51], bins=10)

    assert table["probability_bin"].tolist() == [4, 5]


def test_metrics_by_season_uses_prediction_dates():
    metrics = metrics_by_season(_predictions())

    assert set(metrics) == {"2025", "2026"}
    assert metrics["2025"].n_samples == 2
    assert metrics["2026"].accuracy == pytest.approx(1.0)


def test_metrics_by_season_rejects_missing_seasons():
    with pytest.raises(ValueError, match="valid season"):
        metrics_by_season(pd.DataFrame({"actual_home_win": [1], "home_probability": [0.6], "game_date": ["invalid"]}))
