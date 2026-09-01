"""Small, deterministic baseline for early application development."""
from dataclasses import dataclass
from math import exp
from typing import Sequence

MLB_TEAMS = ["ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WAS", "ATH"]
TEAM_RATINGS = {team: 1500.0 for team in MLB_TEAMS}
HOME_ADVANTAGE = 55.0
LOGISTIC_SCALE = 400.0

@dataclass(frozen=True)
class GamePrediction:
    home_win_probability: float
    away_win_probability: float
    explanation: str

def _logistic(rating_difference: float) -> float:
    return 1.0 / (1.0 + exp(-rating_difference / LOGISTIC_SCALE))

def predict_game(home_team: str, away_team: str, *, home_pitcher: str = "", away_pitcher: str = "", home_lineup: Sequence[str] = (), away_lineup: Sequence[str] = ()) -> GamePrediction:
    """Return a baseline probability using ratings, home advantage, and completeness."""
    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")
    if home_team not in TEAM_RATINGS or away_team not in TEAM_RATINGS:
        raise ValueError("Both teams must be recognized MLB team codes.")
    home_complete = sum(bool(player.strip()) for player in home_lineup)
    away_complete = sum(bool(player.strip()) for player in away_lineup)
    rating_difference = TEAM_RATINGS[home_team] - TEAM_RATINGS[away_team] + HOME_ADVANTAGE + (home_complete - away_complete) * 2.0
    home_probability = _logistic(rating_difference)
    return GamePrediction(home_probability, 1.0 - home_probability, f"Baseline inputs: neutral team ratings, home-field advantage, and lineup completeness ({home_complete}/9 vs {away_complete}/9). Pitcher and player names are not yet statistically modeled.")
