import pandas as pd
import pytest

from src.evaluation.metrics import evaluate_probabilities


def test_evaluation_handles_single_class_validation_split():
    result = evaluate_probabilities(pd.Series([1, 1, 1]), [0.7, 0.8, 0.9])

    assert result.roc_auc is None
    assert result.n_samples == 3
    assert 0 <= result.brier_score <= 1


def test_evaluation_returns_probability_metrics():
    result = evaluate_probabilities(pd.Series([0, 1]), [0.2, 0.8])

    assert result.log_loss < 1
    assert result.brier_score < 0.1
    assert result.roc_auc == 1.0


def test_evaluation_rejects_invalid_probability_inputs():
    with pytest.raises(ValueError, match="probabilities must be between"):
        evaluate_probabilities(pd.Series([0, 1]), [0.2, 1.2])
