"""Model evaluation utilities."""

from .baselines import BaselineResult, elo_baseline, evaluate_baselines, home_team_baseline
from .reporting import calibration_table, metrics_by_season
from .walk_forward import WalkForwardResult, evaluate_walk_forward

__all__ = [
    "BaselineResult",
    "WalkForwardResult",
    "calibration_table",
    "elo_baseline",
    "evaluate_baselines",
    "evaluate_walk_forward",
    "home_team_baseline",
    "metrics_by_season",
]
