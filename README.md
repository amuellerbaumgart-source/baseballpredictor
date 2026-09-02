# MLB Forecast

MLB game win-probability app built with Streamlit, MLB StatsAPI, SQLite, and scikit-learn.

The app has two prediction modes:

- **Early forecast:** works before lineups are announced using season-to-date team records, winning/losing streaks, rest, pitcher availability, and available pitcher statistics such as ERA.
- **Enhanced forecast:** uses the early features plus complete batting orders when lineup data is available.

Predictions are estimates, not guarantees or financial advice.

## 1. Clone the repository

```bash
git clone https://github.com/amuellerbaumgart-source/baseballpredictor.git
cd baseballpredictor
```

## 2. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Run the tests

```bash
pytest
```

The tests use mocked API responses and do not require internet access.

## 5. Start the app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## First-time data setup

The app creates a local SQLite database at `data/mlb.sqlite`.

For a quick start, use the sidebar button **Refresh upcoming games**. This loads the next seven days of games, team names, venues, scheduled times, probable pitchers, and available pitcher ERA.

For model training, run these commands in a second terminal while Streamlit is running:

```bash
source .venv/bin/activate
python -m src.pipeline backfill
python -m src.pipeline backfill-details --limit 500
python -m src.pipeline train
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

What each command does:

1. `backfill` downloads recent historical schedules and completed scores.
2. `backfill-details` downloads completed-game feeds, batting orders, and pitcher statistics. Increase `--limit` to collect more lineup history.
3. `train` creates or updates the early model and creates the enhanced model when at least 30 completed games have complete batting orders.

The early model can predict without lineups. The enhanced model is selected only when both batting orders are complete.

## Daily usage

1. Activate `.venv`.
2. Run `python -m src.pipeline refresh`, or click **Refresh upcoming games** in the app.
3. Select a game in **Matchup forecast**.
4. Refresh pitchers and lineups when MLB publishes new information.
5. Generate the forecast.

The app displays full team names, venue, city, venue-local time, team records, active streaks, probable pitchers, pitcher records, ERA, lineup status, and the selected model.

## Database and generated files

Generated local files are intentionally excluded from Git:

- `data/mlb.sqlite` — local normalized game, lineup, and pitcher-stat database.
- `models/*.joblib` — locally trained model artifacts.

Games use MLB's `game_pk` as the database primary key. Repeating a refresh updates existing games instead of duplicating them. This is idempotent upsert behavior; avoid running multiple simultaneous database-writing commands against the same SQLite file.

## Project structure

```text
app.py                  Streamlit interface
src/data/mlb_api.py     MLB StatsAPI client and response normalization
src/data/ingest.py      Schedule, lineup, and pitcher-stat ingestion
src/data/store.py       SQLite schema, migrations, queries, and upserts
src/models/features.py  Leakage-safe pre-game feature construction
src/models/trainer.py   Chronological training and probability calibration
src/models/predictor.py Model selection and prediction output
src/pipeline.py         Command-line refresh, backfill, and training commands
tests/                  API, database, feature, and leakage tests
```

## Troubleshooting

If the app reports a database lock, stop any other Streamlit or pipeline process using `data/mlb.sqlite`, then restart the app.

If ERA is not displayed, refresh upcoming games or refresh the selected game’s pitcher/lineup data. MLB may not have announced a probable pitcher or may not have returned a season statistic yet.

If the app uses the deterministic baseline, train the early model:

```bash
python -m src.pipeline train
```
