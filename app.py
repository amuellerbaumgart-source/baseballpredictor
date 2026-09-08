"""Streamlit interface for readable MLB forecasts."""
from datetime import date, timedelta
import json
import pandas as pd
import streamlit as st

from src.data.ingest import backfill_historical, refresh_game_details, refresh_games, refresh_probable_pitcher_stats, refresh_teams
from src.data.mlb_api import MLBAPIError, MLBClient, format_pacific_time
from src.data.store import Store
from src.models.features import build_features, build_prediction_features
from src.models.predictor import predict_with_model
from src.models.trainer import train_models

st.set_page_config(page_title="MLB Forecast", page_icon="⚾", layout="wide")
st.markdown("<style>.block-container{max-width:1200px;padding-top:2rem}.hero{padding:1.5rem 2rem;border-radius:16px;background:linear-gradient(120deg,#102a43,#1f5f8b);color:white;margin-bottom:1.5rem}.game-card{padding:1rem;border:1px solid #dbe4ee;border-radius:12px;margin-bottom:1rem}</style>", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>⚾ MLB Forecast</h1><p>Early and lineup-enhanced win probabilities for scheduled MLB games.</p></div>", unsafe_allow_html=True)

store, client = Store(), MLBClient()

def game_label(game):
    return f"{game['away_team_name']} at {game['home_team_name']} · {game['game_date']}"

def feature_input(game):
    home_lineup = json.loads(game["home_lineup_json"] or "[]")
    away_lineup = json.loads(game["away_lineup_json"] or "[]")
    season = int(str(game["game_date"])[:4])
    home_stats, away_stats = store.pitcher_stats(game["home_probable_pitcher_id"], season), store.pitcher_stats(game["away_probable_pitcher_id"], season)
    candidate = dict(game)
    candidate.update({"home_lineup": home_lineup, "away_lineup": away_lineup, "home_pitcher_era": home_stats["era"] if home_stats["available"] else 4.20, "away_pitcher_era": away_stats["era"] if away_stats["available"] else 4.20})
    history = pd.DataFrame([dict(row) for row in store.completed_games()])
    features = build_prediction_features(candidate, history)
    features.update({"home_pitcher": game["home_pitcher_name"] or "", "away_pitcher": game["away_pitcher_name"] or "", "home_lineup": home_lineup, "away_lineup": away_lineup})
    return features

def record_text(record):
    streak = f"{record['streak']} straight" if record["streak"] > 0 else f"{abs(record['streak'])} straight losses" if record["streak"] < 0 else "no active streak"
    return f"{record['wins']}-{record['losses']} · {streak}"

def pitcher_text(name, record, stats):
    if not name: return "Not announced"
    parts = [name]
    if record["available"]: parts.append(f"{record['wins']}-{record['losses']}")
    if stats["available"]: parts.append(f"ERA {stats['era']:.2f}")
    return " · ".join(parts)

with st.sidebar:
    st.header("Data center")
    st.caption(f"Database: `{store.path}`")
    st.metric("Cached games", store.count_games())
    if st.button("Refresh upcoming games", use_container_width=True):
        try:
            refresh_teams(store, client)
            refresh_games(store, client, date.today(), date.today() + timedelta(days=7))
            upcoming = [dict(row) for row in store.upcoming_games(start_date=date.today())]
            stats_count = refresh_probable_pitcher_stats(store, client, upcoming)
            st.success(f"Updated {len(upcoming)} schedule rows and cached {stats_count} pitcher stat records. Existing games were not duplicated.")
        except MLBAPIError as exc: st.error(str(exc))
    if st.button("Backfill last 3 seasons", use_container_width=True):
        try:
            count = backfill_historical(store, client, date(date.today().year - 3, 1, 1), date.today())
            st.success(f"Backfilled {count} rows. Existing game IDs were updated.")
        except MLBAPIError as exc: st.error(str(exc))
    if st.button("Train models", use_container_width=True):
        rows = [dict(row) for row in store.completed_games()]
        if len(rows) < 30: st.warning("Backfill historical data first; at least 30 completed games are required.")
        else:
            try:
                results = train_models(build_features(pd.DataFrame(rows)))
                st.success("Early model trained" + (" and enhanced model trained." if "enhanced" in results else ". Enhanced model needs 30 complete-lineup games."))
            except (ValueError, KeyError) as exc: st.error(str(exc))

tab_schedule, tab_forecast, tab_status = st.tabs(["Upcoming games", "Matchup forecast", "Database & model"])
with tab_schedule:
    st.subheader("Upcoming schedule")
    games = store.upcoming_games(start_date=date.today())
    if not games: st.info("Click Refresh upcoming games to load the schedule.")
    for game in games:
        with st.container(border=True):
            st.markdown(f"### {game['away_team_name']} at {game['home_team_name']}")
            location = ", ".join(x for x in (game["venue_name"], game["venue_city"]) if x)
            pacific_time = format_pacific_time(game["game_datetime"], game["game_date"])
            st.write(" · ".join(x for x in (location, pacific_time, game["status_detail"] or game["status"]) if x))
            season = int(str(game["game_date"])[:4])
            home_record, away_record = store.team_record(game["home_team_id"], season), store.team_record(game["away_team_id"], season)
            st.caption(f"{game['away_team_name']}: {record_text(away_record)} · {game['home_team_name']}: {record_text(home_record)}")
            season = int(str(game["game_date"])[:4])
            away_record, home_record = store.pitcher_record(game["away_probable_pitcher_id"], season), store.pitcher_record(game["home_probable_pitcher_id"], season)
            away_stats, home_stats = store.pitcher_stats(game["away_probable_pitcher_id"], season), store.pitcher_stats(game["home_probable_pitcher_id"], season)
            st.caption(f"Probable pitchers: {pitcher_text(game['away_pitcher_name'], away_record, away_stats)} vs {pitcher_text(game['home_pitcher_name'], home_record, home_stats)} · Lineups: {game['lineup_status']}")
            if st.button("Refresh game details", key=f"details-{game['game_pk']}"):
                try: refresh_game_details(store, client, game["game_pk"]); st.success("Game details updated. Refresh the page to see the latest data.")
                except MLBAPIError as exc: st.error(str(exc))

with tab_forecast:
    st.subheader("Matchup forecast")
    games = store.upcoming_games(start_date=date.today())
    if games:
        options = {game_label(g): g for g in games}
        selected = st.selectbox("Select a game", list(options))
        game = options[selected]
        location = ", ".join(x for x in (game["venue_name"], game["venue_city"]) if x)
        time_text = format_pacific_time(game["game_datetime"], game["game_date"])
        if location or time_text: st.write(" · ".join(x for x in (location, time_text) if x))
        season = int(str(game["game_date"])[:4])
        home_record, away_record = store.team_record(game["home_team_id"], season), store.team_record(game["away_team_id"], season)
        st.caption(f"{game['away_team_name']}: {record_text(away_record)} · {game['home_team_name']}: {record_text(home_record)}")
        left, right = st.columns(2)
        home_lineup, away_lineup = json.loads(game["home_lineup_json"] or "[]"), json.loads(game["away_lineup_json"] or "[]")
        home_pitcher, away_pitcher = store.pitcher_record(game["home_probable_pitcher_id"], season), store.pitcher_record(game["away_probable_pitcher_id"], season)
        left.info(f"{game['home_team_name']} pitcher: {pitcher_text(game['home_pitcher_name'], home_pitcher, store.pitcher_stats(game['home_probable_pitcher_id'], season))}\n\nLineup: {game['lineup_status']}")
        right.info(f"{game['away_team_name']} pitcher: {pitcher_text(game['away_pitcher_name'], away_pitcher, store.pitcher_stats(game['away_probable_pitcher_id'], season))}\n\nLineup: {game['lineup_status']}")
        left.write("Home batting order"); left.write("\n".join(f"{i}. {name}" for i, name in enumerate(home_lineup, 1)) or "Not announced")
        right.write("Away batting order"); right.write("\n".join(f"{i}. {name}" for i, name in enumerate(away_lineup, 1)) or "Not announced")
        if st.button("Refresh pitchers and lineups", key="selected-refresh"):
            try: refresh_game_details(store, client, game["game_pk"]); st.rerun()
            except MLBAPIError as exc: st.error(str(exc))
        if st.button("Generate forecast", type="primary"):
            result, version = predict_with_model(game["home_team_name"], game["away_team_name"], feature_input(game))
            c1, c2 = st.columns(2); c1.metric(f"{game['home_team_name']} win probability", f"{result.home_win_probability:.1%}"); c2.metric(f"{game['away_team_name']} win probability", f"{result.away_win_probability:.1%}")
            st.caption(f"Model: {version} · Lineup state: {game['lineup_status']} · MLB game ID: {game['game_pk']}"); st.info(result.explanation)
    else: st.info("Load upcoming games first.")

with tab_status:
    st.subheader("Database and model status")
    st.write(f"Completed games available for training: **{len(store.completed_games())}**")
    st.write("Every game is keyed by MLB `game_pk`. Refreshing the same date range updates existing games and preserves details; it does not create duplicate games. This is an idempotent upsert, not a multi-process lock.")
    st.write("The early model uses basic team and game context and can predict without a lineup. The enhanced model is selected only when both batting orders are complete and its artifact exists.")
