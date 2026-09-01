# MLB Game Predictor

An evolving Streamlit application for estimating MLB game win probabilities and, later, evaluating player performance and value.

## Current milestone

The app contains a transparent baseline combining neutral team ratings, home-field advantage, and lineup completeness. It is an application scaffold, not a trained predictive model: player and pitcher names do not yet affect the estimate.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Run tests with `pytest`.

## Roadmap

1. Add reliable historical MLB data with pre-game feature snapshots.
2. Train and evaluate chronological baseline models.
3. Add pitcher, bullpen, team, park, and lineup statistics.
4. Add validated player search and role-aware player-value analysis.

Predictions are estimates and should not be treated as guarantees or financial advice.
