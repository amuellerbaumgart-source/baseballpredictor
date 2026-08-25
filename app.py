import streamlit as st

st.title("MLB Game Predictor")

st.divider()

teams = ["ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WAS"]

st.header("Game Information")

home_team = st.selectbox(
    "Home Team",
    teams
)

away_team = st.selectbox(
    "Away Team",
    teams
)

st.divider()

home_pitcher = st.text_input(
    "Home Starting Pitcher"
)

away_pitcher = st.text_input(
    "Away Starting Pitcher"
)

st.divider()

st.subheader("Home Lineup")

home_lineup = []

for i in range(9):
    player = st.text_input(
        f"Home Batter {i+1}"
    )
    home_lineup.append(player)

st.divider()

st.subheader("Away Lineup")

away_lineup = []

for i in range(9):
    player = st.text_input(
        f"Away Batter {i+1}"
    )
    away_lineup.append(player)

st.divider()

predict_button = st.button("Predict")

if predict_button:

    probability = 57.8

    st.subheader("Prediction")

    st.write(
        f"{home_team} Win Probability: {probability}%"
    )
    st.write(
        f"{away_team} Win Probability: {100 - probability}%"
    )
