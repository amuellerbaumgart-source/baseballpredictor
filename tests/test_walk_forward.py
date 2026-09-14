import pandas as pd
import pytest

from src.evaluation.walk_forward import evaluate_walk_forward


def _frame(rows=8):
    return pd.DataFrame(
        {
            "game_pk": range(rows),
            "game_datetime": pd.date_range("2026-04-01", periods=rows, freq="D").astype(str),
            "home_advantage": [1.0] * rows,
            "home_win": [0, 1] * (rows // 2),
        }
    )


def test_walk_forward_uses_only_prior_rows_and_returns_metrics():
    result = evaluate_walk_forward(_frame(), ["home_advantage"], min_train_size=4)

    assert len(result.predictions) == 4
    assert result.predictions["train_rows"].tolist() == [4, 5, 6, 7]
    assert result.metrics.n_samples == 4
    assert result.predictions["game_pk"].tolist() == [4, 5, 6, 7]


def test_walk_forward_sorts_before_splitting():
    frame = _frame().sample(frac=1, random_state=3)

    result = evaluate_walk_forward(frame, ["home_advantage"], min_train_size=4)

    assert result.predictions["game_pk"].tolist() == [4, 5, 6, 7]


def test_walk_forward_requires_features():
    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_walk_forward(_frame(), ["missing_feature"], min_train_size=4)
