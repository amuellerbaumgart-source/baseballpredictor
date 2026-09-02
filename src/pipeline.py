"""Command-line entry points for local refresh and model training."""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from src.data.ingest import backfill_historical, refresh_completed_details, refresh_games, refresh_probable_pitcher_stats, refresh_teams
from src.data.mlb_api import MLBClient
from src.data.store import Store
from src.models.features import build_features
from src.models.trainer import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh MLB data or train the win model")
    parser.add_argument("action", choices=["refresh", "backfill", "backfill-details", "train"])
    parser.add_argument("--limit", type=int, default=500, help="Maximum completed game feeds to fetch for backfill-details")
    parser.add_argument("--days", type=int, default=7, help="Number of future days to refresh")
    args = parser.parse_args()
    store = Store()
    if args.action == "refresh":
        client = MLBClient()
        refresh_teams(store, client)
        refresh_games(store, client, date.today(), date.today() + timedelta(days=args.days))
        count = refresh_probable_pitcher_stats(store, client, [dict(row) for row in store.upcoming_games()])
        print(f"Refreshed upcoming schedule and cached {count} pitcher stat records.")
        return
    if args.action == "backfill":
        client = MLBClient()
        refresh_teams(store, client)
        start = date(date.today().year - 3, 1, 1)
        count = backfill_historical(store, client, start, date.today())
        print(f"Backfilled {count} schedule rows; database total is {store.count_games()} unique games.")
        return
    if args.action == "backfill-details":
        count = refresh_completed_details(store, MLBClient(), args.limit)
        print(f"Fetched details for {count} completed games.")
        return
    rows = [dict(row) for row in store.completed_games()]
    metrics = train_models(build_features(pd.DataFrame(rows)))
    print(metrics)


if __name__ == "__main__":
    main()
