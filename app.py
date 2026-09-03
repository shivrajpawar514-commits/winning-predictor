import streamlit as st
import joblib
import numpy as np

# Load models and encoders
clf = joblib.load('ipl_win_predictor.pkl')
reg = joblib.load('ipl_runs_predictor.pkl')
le_batting = joblib.load('le_batting.pkl')
le_bowling = joblib.load('le_bowling.pkl')

st.title("🏏 IPL Match Predictor")
st.write("Predict the winning probability and total runs during an IPL match.")

# Select innings
innings = st.radio("Select Innings", options=[1, 2])

# Select teams - using exact classes from label encoders
teams = list(le_batting.classes_)

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)

if batting_team == bowling_team:
    st.warning("Batting and bowling teams cannot be the same.")
    st.stop()

# Inputs for match state
runs_scored = st.number_input("Runs scored so far", min_value=0, max_value=500, value=50)
wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=10, value=2)
overs_completed = st.number_input("Overs completed (e.g. 7.2 for 7 overs 2 balls)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)

# Convert overs like 7.2 (7 overs and 2 balls) to decimal overs
balls = int((overs_completed - int(overs_completed)) * 10)
overs_decimal = int(overs_completed) + balls / 6

# Target runs input only for 2nd innings
target_runs = 0
if innings == 2:
    target_runs = st.number_input("Target runs to chase", min_value=1, max_value=500, value=150)

if st.button("Predict"):
    try:
        batting_enc = le_batting.transform([batting_team])[0]
        bowling_enc = le_bowling.transform([bowling_team])[0]
    except ValueError:
        st.error("Teams not recognized by the model.")
        st.stop()

    if innings == 1:
        st.subheader("🏆 Win Probability")
        st.write("Win probability prediction is **not available for the first innings**. It is typically calculated in the second innings.")
    else:
        # Prepare feature vector for classification model
        features_cls = np.array([[batting_enc, bowling_enc, runs_scored, wickets_lost, overs_decimal, innings, target_runs]])
        
        win_prob = clf.predict_proba(features_cls)[0][1]

        st.subheader("🏆 Win Probability")
        st.write(f"Probability that **{batting_team}** will win: **{win_prob * 100:.2f}%**")
        st.progress(win_prob)

        # Simple interpretation
        if win_prob > 0.7:
            st.success(f"{batting_team} are favorites to win! 🎉")
        elif win_prob > 0.4:
            st.info("Match is balanced! 🤝")
        else:
            st.error(f"{batting_team} is unlikely to win based on current stats. 😞")

    if innings == 1:
        # Predict total runs by innings end
        features_reg = np.array([[batting_enc, bowling_enc, runs_scored, wickets_lost, overs_decimal]])
        predicted_total_runs = reg.predict(features_reg)[0]
        st.subheader("📊 Predicted Total Runs by Innings End")
        st.write(f"Predicted total runs: **{predicted_total_runs:.1f}**")

st.markdown("---")
st.caption("Model trained on IPL deliveries and match data. Predictions update as match progresses.")

# Footer
st.markdown(
    """
    ---
    Created with ❤️ for IPL analytics by Shivraj.
    """
)
