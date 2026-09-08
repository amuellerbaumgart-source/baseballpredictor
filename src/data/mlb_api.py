"""Small, defensive client for MLB's StatsAPI."""
from __future__ import annotations

from datetime import date, timedelta
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

import requests

PACIFIC_TIMEZONE = ZoneInfo("America/Los_Angeles")


def format_pacific_time(game_datetime: str | None, fallback: str = "") -> str:
    """Format an ISO game timestamp in Pacific time for display."""
    if not game_datetime:
        return fallback
    try:
        timestamp = datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
        return timestamp.astimezone(PACIFIC_TIMEZONE).strftime("%Y-%m-%d %I:%M %p PT")
    except (AttributeError, TypeError, ValueError):
        return fallback or game_datetime


class MLBAPIError(RuntimeError):
    """Raised when MLB StatsAPI cannot provide a valid response."""


class MLBClient:
    def __init__(self, base_url: str = "https://statsapi.mlb.com/api/v1", timeout: int = 15, session=None):
        self.base_url = base_url.rstrip("/")
        self.live_base_url = self.base_url.replace("/v1", "/v1.1", 1)
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get(self, path: str, base_url: str | None = None, **params: Any) -> dict[str, Any]:
        try:
            response = self.session.get(f"{(base_url or self.base_url)}/{path.lstrip('/')}", params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MLBAPIError(f"MLB API request failed for {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MLBAPIError(f"MLB API returned an unexpected response for {path}")
        return payload

    def teams(self) -> list[dict[str, Any]]:
        return self._get("teams", sportId=1).get("teams", [])

    def schedule(self, start_date: date, end_date: date | None = None) -> list[dict[str, Any]]:
        end_date = end_date or start_date
        return self._get("schedule", sportId=1, startDate=start_date.isoformat(), endDate=end_date.isoformat(), hydrate="team,probablePitcher,venue").get("dates", [])

    def games(self, start_date: date, end_date: date | None = None) -> list[dict[str, Any]]:
        return [game for day in self.schedule(start_date, end_date) for game in day.get("games", [])]

    @staticmethod
    def normalize_game(game: dict[str, Any]) -> dict[str, Any]:
        teams = game.get("teams", {})
        home, away = teams.get("home", {}), teams.get("away", {})
        venue = game.get("venue", {})
        timezone_value = venue.get("timeZone", "UTC")
        if isinstance(timezone_value, dict):
            timezone = timezone_value.get("id") or timezone_value.get("tz") or "UTC"
        elif isinstance(timezone_value, str) and timezone_value:
            timezone = timezone_value
        else:
            timezone = "UTC"
        scheduled = game.get("gameDate")
        local_time = ""
        if scheduled:
            try:
                local_time = datetime.fromisoformat(scheduled.replace("Z", "+00:00")).astimezone(ZoneInfo(timezone)).isoformat()
            except (ValueError, KeyError, TypeError):
                local_time = scheduled
        return {"game_pk": game.get("gamePk"), "game_date": game.get("officialDate", ""), "game_datetime": scheduled, "scheduled_local_time": local_time, "game_type": game.get("gameType", "R"), "status": game.get("status", {}).get("abstractGameState", "Unknown"), "status_detail": game.get("status", {}).get("detailedState", ""), "home_team_id": home.get("team", {}).get("id"), "away_team_id": away.get("team", {}).get("id"), "home_team_code": home.get("team", {}).get("abbreviation", ""), "away_team_code": away.get("team", {}).get("abbreviation", ""), "home_team_name": home.get("team", {}).get("name", "Home team"), "away_team_name": away.get("team", {}).get("name", "Away team"), "home_score": home.get("score"), "away_score": away.get("score"), "venue_name": venue.get("name", ""), "venue_city": venue.get("location", {}).get("city", ""), "venue_timezone": timezone, "home_probable_pitcher_id": home.get("probablePitcher", {}).get("id"), "away_probable_pitcher_id": away.get("probablePitcher", {}).get("id"), "home_pitcher_name": home.get("probablePitcher", {}).get("fullName", ""), "away_pitcher_name": away.get("probablePitcher", {}).get("fullName", "")}

    def game_feed(self, game_pk: int) -> dict[str, Any]:
        return self._get(f"game/{game_pk}/feed/live", base_url=self.live_base_url)

    @staticmethod
    def extract_game_details(feed: dict[str, Any]) -> dict[str, Any]:
        """Extract probable pitchers and batting order from a live game feed."""
        game_data = feed.get("gameData", {})
        probable = game_data.get("probablePitchers", {})
        live_teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
        details = {"home_pitcher_id": probable.get("home", {}).get("id"), "away_pitcher_id": probable.get("away", {}).get("id"), "home_pitcher_name": probable.get("home", {}).get("fullName", ""), "away_pitcher_name": probable.get("away", {}).get("fullName", ""), "home_lineup": [], "away_lineup": []}
        for side in ("home", "away"):
            players = live_teams.get(side, {}).get("players", {})
            ordered = []
            for player in players.values():
                order = player.get("battingOrder")
                if order is not None:
                    ordered.append((int(order), player.get("person", {}).get("fullName", "")))
            details[f"{side}_lineup"] = [name for _, name in sorted(ordered) if name]
        return details

    def person_stats(self, person_id: int, season: int) -> dict[str, Any]:
        return self._get(f"people/{person_id}/stats", stats=",".join(["season"]), group="hitting,pitching", season=season)

    def pitcher_season_stats(self, person_id: int, season: int) -> dict[str, Any]:
        payload = self._get(f"people/{person_id}/stats", stats="season", group="pitching", season=season)
        for block in payload.get("stats", []):
            for split in block.get("splits", []):
                stat = split.get("stat", {})
                if stat.get("era") is not None:
                    return {"pitcher_id": person_id, "season": season, "era": float(stat["era"]), "wins": int(stat.get("wins", 0)), "losses": int(stat.get("losses", 0))}
        return {"pitcher_id": person_id, "season": season, "era": None, "wins": 0, "losses": 0}

    def team_stats(self, team_id: int, season: int) -> dict[str, Any]:
        return self._get(f"teams/{team_id}/stats", stats="season", group="hitting,pitching", season=season)

    def recent_dates(self, seasons: int = 3) -> tuple[date, date]:
        end = date.today()
        return end - timedelta(days=365 * seasons), end
