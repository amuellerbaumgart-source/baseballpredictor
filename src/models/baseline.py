"""Small, deterministic baseline for early application development."""
from dataclasses import dataclass
from math import exp
from typing import Sequence

MLB_TEAMS = ["ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WAS", "ATH"]
TEAM_RATINGS = {team: 1500.0 for team in MLB_TEAMS}
HOME_ADVANTAGE = 55.0
LOGISTIC_SCALE = 400.0
TEAM_NAMES = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "Seattle Mariners": "SEA", "San Francisco Giants": "SF",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS", "Athletics": "ATH",
}


def normalize_team_identifier(identifier: str) -> str:
    """Return the internal MLB code for a display name or code."""
    normalized = str(identifier).strip()
    if normalized.upper() in TEAM_RATINGS:
        return normalized.upper()
    names = {name.casefold(): code for name, code in TEAM_NAMES.items()}
    try:
        return names[normalized.casefold()]
    except KeyError as exc:
        raise ValueError(f"Unrecognized MLB team identifier: {identifier}") from exc

@dataclass(frozen=True)
class GamePrediction:
    home_win_probability: float
    away_win_probability: float
    explanation: str

def _logistic(rating_difference: float) -> float:
    return 1.0 / (1.0 + exp(-rating_difference / LOGISTIC_SCALE))

def predict_game(home_team: str, away_team: str, *, home_pitcher: str = "", away_pitcher: str = "", home_lineup: Sequence[str] = (), away_lineup: Sequence[str] = ()) -> GamePrediction:
    """Return a baseline probability using ratings, home advantage, and completeness."""
    home_code = normalize_team_identifier(home_team)
    away_code = normalize_team_identifier(away_team)
    if home_code == away_code:
        raise ValueError("Home and away teams must be different.")
    home_complete = sum(bool(player.strip()) for player in home_lineup)
    away_complete = sum(bool(player.strip()) for player in away_lineup)
    rating_difference = TEAM_RATINGS[home_code] - TEAM_RATINGS[away_code] + HOME_ADVANTAGE + (home_complete - away_complete) * 2.0
    home_probability = _logistic(rating_difference)
    return GamePrediction(home_probability, 1.0 - home_probability, f"Baseline inputs: neutral team ratings, home-field advantage, and lineup completeness ({home_complete}/9 vs {away_complete}/9). Pitcher and player names are not yet statistically modeled.")
