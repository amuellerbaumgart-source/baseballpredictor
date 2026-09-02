"""SQLite persistence for normalized games and prediction snapshots."""
from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (team_id INTEGER PRIMARY KEY, code TEXT UNIQUE NOT NULL, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS games (game_pk INTEGER PRIMARY KEY, game_date TEXT NOT NULL, game_datetime TEXT, scheduled_local_time TEXT, game_type TEXT NOT NULL DEFAULT 'R', status TEXT, status_detail TEXT, home_team_id INTEGER, away_team_id INTEGER, home_team_code TEXT, away_team_code TEXT, home_team_name TEXT, away_team_name TEXT, home_score INTEGER, away_score INTEGER, venue_name TEXT, venue_city TEXT, venue_timezone TEXT, home_probable_pitcher_id INTEGER, away_probable_pitcher_id INTEGER, home_pitcher_name TEXT, away_pitcher_name TEXT, home_lineup_json TEXT NOT NULL DEFAULT '[]', away_lineup_json TEXT NOT NULL DEFAULT '[]', lineup_status TEXT NOT NULL DEFAULT 'Not available', updated_at TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS feature_snapshots (game_pk INTEGER PRIMARY KEY, created_at TEXT NOT NULL, feature_json TEXT NOT NULL, home_win INTEGER, model_version TEXT);
CREATE TABLE IF NOT EXISTS predictions (game_pk INTEGER PRIMARY KEY, created_at TEXT NOT NULL, home_probability REAL NOT NULL, away_probability REAL NOT NULL, model_version TEXT NOT NULL, explanation TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pitcher_season_stats (pitcher_id INTEGER NOT NULL, season INTEGER NOT NULL, era REAL, wins INTEGER NOT NULL DEFAULT 0, losses INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (pitcher_id, season));
"""


class Store:
    def __init__(self, path: str = "data/mlb.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(games)")}
        added_game_type = "game_type" not in columns
        for name, definition in (("scheduled_local_time", "TEXT"), ("game_type", "TEXT NOT NULL DEFAULT 'R'"), ("status_detail", "TEXT"), ("home_team_code", "TEXT"), ("away_team_code", "TEXT"), ("home_team_name", "TEXT"), ("away_team_name", "TEXT"), ("venue_name", "TEXT"), ("venue_city", "TEXT"), ("venue_timezone", "TEXT"), ("home_pitcher_name", "TEXT"), ("away_pitcher_name", "TEXT"), ("home_lineup_json", "TEXT NOT NULL DEFAULT '[]'"), ("away_lineup_json", "TEXT NOT NULL DEFAULT '[]'"), ("lineup_status", "TEXT NOT NULL DEFAULT 'Not available'"), ("updated_at", "TEXT NOT NULL DEFAULT ''")):
            if name not in columns:
                connection.execute(f"ALTER TABLE games ADD COLUMN {name} {definition}")
        if added_game_type:
            connection.execute("UPDATE games SET game_type='S' WHERE substr(game_date, 6, 2)='03'")
        legacy_columns = {row[1] for row in connection.execute("PRAGMA table_info(games)")}
        if "venue" in legacy_columns:
            connection.execute("UPDATE games SET venue_name=COALESCE(NULLIF(venue_name, ''), venue) WHERE venue_name IS NULL OR venue_name = ''")
        connection.execute("UPDATE games SET home_team_name=(SELECT name FROM teams WHERE teams.team_id=games.home_team_id) WHERE (home_team_name IS NULL OR home_team_name='') AND home_team_id IS NOT NULL")
        connection.execute("UPDATE games SET away_team_name=(SELECT name FROM teams WHERE teams.team_id=games.away_team_id) WHERE (away_team_name IS NULL OR away_team_name='') AND away_team_id IS NOT NULL")
        connection.commit()
        return connection

    def upsert_teams(self, teams: Iterable[dict]) -> None:
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO teams(team_id, code, name) VALUES (?, ?, ?)", [(t["id"], t.get("abbreviation", t.get("teamName", "")), t["name"]) for t in teams])

    def upsert_games(self, games: Iterable[dict]) -> None:
        rows = []
        for game in games:
            item = game if "game_pk" in game else {"game_pk": game.get("gamePk")}
            if "game_pk" not in game:
                from .mlb_api import MLBClient
                item = MLBClient.normalize_game(game)
            rows.append(tuple(item.get(key) for key in ("game_pk", "game_date", "game_datetime", "scheduled_local_time", "game_type", "status", "status_detail", "home_team_id", "away_team_id", "home_team_code", "away_team_code", "home_team_name", "away_team_name", "home_score", "away_score", "venue_name", "venue_city", "venue_timezone", "home_probable_pitcher_id", "away_probable_pitcher_id", "home_pitcher_name", "away_pitcher_name")) + ("[]", "[]", "Not available", datetime.now(timezone.utc).isoformat()))
        with self.connect() as db:
            columns = "game_pk,game_date,game_datetime,scheduled_local_time,game_type,status,status_detail,home_team_id,away_team_id,home_team_code,away_team_code,home_team_name,away_team_name,home_score,away_score,venue_name,venue_city,venue_timezone,home_probable_pitcher_id,away_probable_pitcher_id,home_pitcher_name,away_pitcher_name,home_lineup_json,away_lineup_json,lineup_status,updated_at"
            db.executemany(f"INSERT INTO games({columns}) VALUES ({','.join('?' for _ in range(26) )}) ON CONFLICT(game_pk) DO UPDATE SET game_date=excluded.game_date,game_datetime=excluded.game_datetime,scheduled_local_time=excluded.scheduled_local_time,game_type=excluded.game_type,status=excluded.status,status_detail=excluded.status_detail,home_team_id=excluded.home_team_id,away_team_id=excluded.away_team_id,home_team_code=excluded.home_team_code,away_team_code=excluded.away_team_code,home_team_name=excluded.home_team_name,away_team_name=excluded.away_team_name,home_score=excluded.home_score,away_score=excluded.away_score,venue_name=excluded.venue_name,venue_city=excluded.venue_city,venue_timezone=excluded.venue_timezone,home_probable_pitcher_id=COALESCE(excluded.home_probable_pitcher_id,games.home_probable_pitcher_id),away_probable_pitcher_id=COALESCE(excluded.away_probable_pitcher_id,games.away_probable_pitcher_id),home_pitcher_name=COALESCE(NULLIF(excluded.home_pitcher_name,''),games.home_pitcher_name),away_pitcher_name=COALESCE(NULLIF(excluded.away_pitcher_name,''),games.away_pitcher_name),updated_at=excluded.updated_at", rows)

    def upsert_game_details(self, game_pk: int, details: dict) -> None:
        with self.connect() as db:
            home, away = details.get("home_lineup", []), details.get("away_lineup", [])
            status = "Confirmed" if len(home) >= 9 and len(away) >= 9 else "Partial" if home or away else "Not available"
            db.execute("UPDATE games SET home_probable_pitcher_id=COALESCE(?, home_probable_pitcher_id), away_probable_pitcher_id=COALESCE(?, away_probable_pitcher_id), home_pitcher_name=COALESCE(NULLIF(?, ''), home_pitcher_name), away_pitcher_name=COALESCE(NULLIF(?, ''), away_pitcher_name), home_lineup_json=?, away_lineup_json=?, lineup_status=?, updated_at=? WHERE game_pk=?", (details.get("home_pitcher_id"), details.get("away_pitcher_id"), details.get("home_pitcher_name", ""), details.get("away_pitcher_name", ""), json.dumps(home), json.dumps(away), status, datetime.now(timezone.utc).isoformat(), game_pk))

    def game_season(self, game_pk: int) -> int:
        with self.connect() as db:
            row = db.execute("SELECT substr(game_date, 1, 4) FROM games WHERE game_pk=?", (game_pk,)).fetchone()
        return int(row[0]) if row and row[0] else datetime.now(timezone.utc).year

    def upsert_pitcher_stats(self, stats: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO pitcher_season_stats(pitcher_id, season, era, wins, losses, updated_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(pitcher_id, season) DO UPDATE SET era=excluded.era, wins=excluded.wins, losses=excluded.losses, updated_at=excluded.updated_at", (stats["pitcher_id"], stats["season"], stats.get("era"), stats.get("wins", 0), stats.get("losses", 0), datetime.now(timezone.utc).isoformat()))

    def pitcher_stats(self, pitcher_id: int | None, season: int | None = None) -> dict:
        if not pitcher_id: return {"era": None, "wins": 0, "losses": 0, "available": False}
        season = season or datetime.now(timezone.utc).year
        with self.connect() as db:
            row = db.execute("SELECT era, wins, losses FROM pitcher_season_stats WHERE pitcher_id=? AND season=?", (pitcher_id, season)).fetchone()
        if row is None:
            return {"era": None, "wins": 0, "losses": 0, "available": False}
        return {"era": row["era"], "wins": row["wins"], "losses": row["losses"], "available": row["era"] is not None}

    def upcoming_games(self, limit: int = 100):
        with self.connect() as db:
            return db.execute("SELECT * FROM games WHERE status != 'Final' ORDER BY game_datetime LIMIT ?", (limit,)).fetchall()

    def completed_games(self):
        with self.connect() as db:
            return db.execute("SELECT games.*, hs.era AS home_pitcher_era, hs.wins AS home_pitcher_stat_wins, hs.losses AS home_pitcher_stat_losses, aws.era AS away_pitcher_era, aws.wins AS away_pitcher_stat_wins, aws.losses AS away_pitcher_stat_losses FROM games LEFT JOIN pitcher_season_stats hs ON hs.pitcher_id=games.home_probable_pitcher_id AND hs.season=substr(games.game_date,1,4) LEFT JOIN pitcher_season_stats aws ON aws.pitcher_id=games.away_probable_pitcher_id AND aws.season=substr(games.game_date,1,4) WHERE status = 'Final' AND game_type = 'R' AND home_score IS NOT NULL AND away_score IS NOT NULL ORDER BY game_date").fetchall()

    def count_games(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM games").fetchone()[0])

    def model_ready(self, minimum_games: int = 30) -> bool:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM games WHERE status = 'Final' AND home_score IS NOT NULL AND away_score IS NOT NULL").fetchone()[0]) >= minimum_games

    def team_record(self, team_id: int, season: int | None = None) -> dict:
        """Return a season-to-date record and active streak."""
        season = season or datetime.now(timezone.utc).year
        with self.connect() as db:
            games = db.execute("SELECT home_team_id, away_team_id, home_score, away_score FROM games WHERE status = 'Final' AND (game_type = 'R' OR (game_type IS NULL AND substr(game_date, 6, 2) >= '04')) AND home_score IS NOT NULL AND away_score IS NOT NULL AND substr(game_date, 1, 4) = ? ORDER BY game_date, game_datetime", (str(season),)).fetchall()
        wins = losses = streak = 0
        for game in games:
            if game["home_team_id"] != team_id and game["away_team_id"] != team_id: continue
            won = (game["home_team_id"] == team_id and game["home_score"] > game["away_score"]) or (game["away_team_id"] == team_id and game["away_score"] > game["home_score"])
            if won:
                wins += 1; streak = streak + 1 if streak >= 0 else 1
            else:
                losses += 1; streak = streak - 1 if streak <= 0 else -1
        return {"wins": wins, "losses": losses, "streak": streak}

    def pitcher_record(self, pitcher_id: int | None, season: int | None = None) -> dict:
        """Return a season-to-date record for a probable/starting pitcher."""
        season = season or datetime.now(timezone.utc).year
        if not pitcher_id:
            return {"wins": 0, "losses": 0, "streak": 0, "available": False}
        with self.connect() as db:
            games = db.execute("SELECT home_probable_pitcher_id, away_probable_pitcher_id, home_score, away_score FROM games WHERE status = 'Final' AND (game_type = 'R' OR (game_type IS NULL AND substr(game_date, 6, 2) >= '04')) AND home_score IS NOT NULL AND away_score IS NOT NULL AND substr(game_date, 1, 4) = ? AND (home_probable_pitcher_id = ? OR away_probable_pitcher_id = ?) ORDER BY game_date, game_datetime", (str(season), pitcher_id, pitcher_id)).fetchall()
        wins = losses = streak = 0
        for game in games:
            home = game["home_probable_pitcher_id"] == pitcher_id
            won = game["home_score"] > game["away_score"] if home else game["away_score"] > game["home_score"]
            if won:
                wins += 1; streak = streak + 1 if streak >= 0 else 1
            else:
                losses += 1; streak = streak - 1 if streak <= 0 else -1
        return {"wins": wins, "losses": losses, "streak": streak, "available": True}
