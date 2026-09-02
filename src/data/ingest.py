"""Orchestrates StatsAPI calls into the local store."""
from __future__ import annotations

from datetime import date, timedelta

from .mlb_api import MLBAPIError, MLBClient
from .store import Store


def refresh_games(store: Store, client: MLBClient, start_date: date, end_date: date | None = None) -> int:
    games = client.games(start_date, end_date)
    store.upsert_games(games)
    return len(games)


def refresh_teams(store: Store, client: MLBClient) -> int:
    teams = client.teams()
    store.upsert_teams(teams)
    return len(teams)


def refresh_game_details(store: Store, client: MLBClient, game_pk: int) -> None:
    details = client.extract_game_details(client.game_feed(game_pk))
    store.upsert_game_details(game_pk, details)
    season = store.game_season(game_pk)
    for pitcher_id in (details.get("home_pitcher_id"), details.get("away_pitcher_id")):
        if pitcher_id:
            try: store.upsert_pitcher_stats(client.pitcher_season_stats(pitcher_id, season))
            except MLBAPIError: pass


def refresh_probable_pitcher_stats(store: Store, client: MLBClient, games) -> int:
    """Cache season stats for announced probable pitchers without requiring lineups."""
    fetched = 0
    seen = set()
    for game in games:
        season = int(str(game.get("game_date", ""))[:4])
        for pitcher_id in (game.get("home_probable_pitcher_id"), game.get("away_probable_pitcher_id")):
            if not pitcher_id or (pitcher_id, season) in seen: continue
            seen.add((pitcher_id, season))
            try:
                store.upsert_pitcher_stats(client.pitcher_season_stats(pitcher_id, season))
                fetched += 1
            except MLBAPIError:
                continue
    return fetched


def backfill_historical(store: Store, client: MLBClient, start_date: date, end_date: date) -> int:
    """Backfill completed schedule rows in monthly chunks; repeated calls are safe."""
    total = 0
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor.replace(day=28) + timedelta(days=4))
        chunk_end = chunk_end.replace(day=1) - timedelta(days=1) if chunk_end.month != cursor.month else chunk_end
        total += refresh_games(store, client, cursor, chunk_end)
        cursor = chunk_end + timedelta(days=1)
    return total


def refresh_completed_details(store: Store, client: MLBClient, limit: int = 500) -> int:
    """Fetch box scores for completed games so batting orders can be modeled."""
    count = 0
    for game in store.completed_games()[-limit:]:
        refresh_game_details(store, client, game["game_pk"])
        count += 1
    return count
