"""Metrics for probability forecasts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class Evaluation:
    log_loss: float
    brier_score: float
    accuracy: float
    roc_auc: float | None
    n_samples: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def evaluate_probabilities(actual: pd.Series, probabilities) -> Evaluation:
    """Evaluate probabilities without failing when a split has one class."""
    actual = pd.Series(actual).astype(int)
    probabilities = pd.Series(probabilities, dtype=float)
    if actual.empty or len(actual) != len(probabilities):
        raise ValueError("actual and probabilities must have the same non-zero length.")
    if not actual.isin([0, 1]).all() or not probabilities.map(math.isfinite).all():
        raise ValueError("actual must contain only 0/1 values and probabilities must be finite.")
    if not probabilities.between(0.0, 1.0).all():
        raise ValueError("probabilities must be between 0 and 1.")
    probabilities = probabilities.clip(1e-6, 1 - 1e-6)
    roc_auc = float(roc_auc_score(actual, probabilities)) if actual.nunique() == 2 else None
    return Evaluation(
        log_loss=float(log_loss(actual, probabilities, labels=[0, 1])),
        brier_score=float(brier_score_loss(actual, probabilities)),
        accuracy=float(accuracy_score(actual, probabilities >= 0.5)),
        roc_auc=roc_auc,
        n_samples=len(actual),
    )
