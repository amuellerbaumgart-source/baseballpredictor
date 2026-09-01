"""Streamlit interface for the MLB game prediction baseline."""
import streamlit as st
from src.models.baseline import MLB_TEAMS, predict_game

st.set_page_config(page_title="MLB Game Predictor", page_icon="⚾")
st.title("MLB Game Predictor")
st.caption("A transparent baseline model. Probabilities are estimates, not guarantees.")
st.header("Game information")
home_team = st.selectbox("Home team", MLB_TEAMS, index=MLB_TEAMS.index("LAD"))
away_team = st.selectbox("Away team", MLB_TEAMS, index=MLB_TEAMS.index("NYY"))
home_pitcher = st.text_input("Home starting pitcher (optional)")
away_pitcher = st.text_input("Away starting pitcher (optional)")
st.header("Lineup information")
st.write("The current baseline uses lineup completeness. Player-level statistics will be added in the next milestone.")
home_lineup = [st.text_input(f"Home batter {i + 1}", key=f"home-{i}") for i in range(9)]
away_lineup = [st.text_input(f"Away batter {i + 1}", key=f"away-{i}") for i in range(9)]

if st.button("Predict", type="primary"):
    if home_team == away_team:
        st.error("Choose two different teams.")
    else:
        result = predict_game(home_team, away_team, home_pitcher=home_pitcher, away_pitcher=away_pitcher, home_lineup=home_lineup, away_lineup=away_lineup)
        st.subheader("Prediction")
        left, right = st.columns(2)
        left.metric(f"{home_team} win probability", f"{result.home_win_probability:.1%}")
        right.metric(f"{away_team} win probability", f"{result.away_win_probability:.1%}")
        st.info(result.explanation)
