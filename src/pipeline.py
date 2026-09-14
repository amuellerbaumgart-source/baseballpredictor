"""Command-line entry points for local refresh and model training."""
from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from src.data.ingest import (
    backfill_historical,
    refresh_completed_details,
    refresh_games,
    refresh_probable_pitcher_stats,
    refresh_teams,
)
from src.data.mlb_api import MLBClient
from src.data.store import Store
from src.evaluation.baselines import evaluate_baselines
from src.evaluation.reporting import calibration_table, metrics_by_season
from src.evaluation.walk_forward import evaluate_walk_forward
from src.models.features import EARLY_FEATURE_COLUMNS, build_features
from src.models.trainer import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh MLB data or train the win model")
    parser.add_argument("action", choices=["refresh", "backfill", "backfill-details", "train", "evaluate"])
    parser.add_argument("--limit", type=int, default=500, help="Maximum completed game feeds to fetch for backfill-details")
    parser.add_argument("--days", type=int, default=7, help="Number of future days to refresh")
    args = parser.parse_args()
    store = Store()
    if args.action == "refresh":
        today = datetime.now(UTC).date()
        client = MLBClient()
        refresh_teams(store, client)
        refresh_games(store, client, today, today + timedelta(days=args.days))
        count = refresh_probable_pitcher_stats(store, client, [dict(row) for row in store.upcoming_games(start_date=today)])
        print(f"Refreshed upcoming schedule and cached {count} pitcher stat records.")
        return
    if args.action == "backfill":
        today = datetime.now(UTC).date()
        client = MLBClient()
        refresh_teams(store, client)
        start = date(today.year - 3, 1, 1)
        count = backfill_historical(store, client, start, today)
        print(f"Backfilled {count} schedule rows; database total is {store.count_games()} unique games.")
        return
    if args.action == "backfill-details":
        count = refresh_completed_details(store, MLBClient(), args.limit)
        print(f"Fetched details for {count} completed games.")
        return
    rows = [dict(row) for row in store.completed_games()]
    source_frame = pd.DataFrame(rows)
    features = build_features(source_frame)
    if args.action == "evaluate":
        result = evaluate_walk_forward(features, EARLY_FEATURE_COLUMNS)
        baseline_frame = source_frame[["game_pk", "game_date", "game_datetime", "home_team_id", "away_team_id"]].merge(
            features[["game_pk", "home_win"]], on="game_pk", how="inner"
        )
        output = {
            "walk_forward_model": result.metrics.to_dict(),
            "calibration": calibration_table(
                result.predictions["actual_home_win"], result.predictions["home_probability"]
            ).to_dict(orient="records"),
            "by_season": {season: metrics.to_dict() for season, metrics in metrics_by_season(result.predictions).items()},
        }
        output["baselines"] = {
            name: benchmark.metrics.to_dict() for name, benchmark in evaluate_baselines(baseline_frame, test_start=30).items()
        }
        print(output)
        return
    metrics = train_models(features)
    print(metrics)


if __name__ == "__main__":
    main()
