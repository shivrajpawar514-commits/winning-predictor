import streamlit as st
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="IPL Match Predictor | Win & Score Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #FF4655 0%, #FFAE34 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.25rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .prob-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .team-badge {
        font-weight: 700;
        font-size: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model Loading for instant responsiveness
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    clf = joblib.load('ipl_win_predictor.pkl')
    reg = joblib.load('ipl_runs_predictor.pkl')
    le_batting = joblib.load('le_batting.pkl')
    le_bowling = joblib.load('le_bowling.pkl')
    return clf, reg, le_batting, le_bowling

try:
    clf, reg, le_batting, le_bowling = load_models()
    teams = sorted(list(le_batting.classes_))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Header
st.markdown('<h1 class="main-title">🏏 IPL Match Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time Machine Learning Win Probability & Score Projection Engine</p>', unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Match Configuration")
    innings = st.radio("Select Current Innings", options=[1, 2], format_func=lambda x: f"Innings {x} ({'1st Innings: Total Runs Projection' if x == 1 else '2nd Innings: Chase Win Probability'})")
    st.markdown("---")
    st.markdown("### 📊 About the Model")
    st.info("Powered by **Random Forest Ensemble Models** trained on historical ball-by-ball IPL match data (2008-2024).")
    st.caption("Developed by **Shivraj**")

# Main Input Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏏 Batting Team")
    batting_team = st.selectbox("Select Batting Team", teams, index=0, key="batting")

with col2:
    st.markdown("### 🎯 Bowling Team")
    bowling_team = st.selectbox("Select Bowling Team", teams, index=1 if len(teams) > 1 else 0, key="bowling")

if batting_team == bowling_team:
    st.warning("⚠️ Batting and bowling teams cannot be the same. Please choose distinct teams.")
    st.stop()

st.markdown("---")
st.markdown("### ⏱️ Live Match Situation")

col_runs, col_wickets, col_overs = st.columns(3)

with col_runs:
    runs_scored = st.number_input("Runs Scored", min_value=0, max_value=500, value=65, step=1, help="Current total runs scored by the batting team.")

with col_wickets:
    wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2, step=1, help="Wickets fallen so far in this innings.")

with col_overs:
    overs_completed = st.number_input("Overs Completed", min_value=0.1, max_value=20.0, value=7.3, step=0.1, help="Format: e.g. 7.3 for 7 overs and 3 balls")

# Calculate decimal overs and balls
overs_int = int(overs_completed)
balls = int(round((overs_completed - overs_int) * 10))
if balls > 5:
    balls = 5
overs_decimal = overs_int + (balls / 6.0)
balls_bowled = int(overs_int * 6 + balls)
balls_remaining = max(0, 120 - balls_bowled)

# Target runs input for 2nd innings
target_runs = 0
if innings == 2:
    col_target, col_empty = st.columns([1, 1])
    with col_target:
        target_runs = st.number_input("Target Score to Chase", min_value=1, max_value=500, value=175, step=1, help="Total runs required to win the match.")

# Live Metrics Dashboard
st.markdown("#### 📈 Current Match Metrics")
m1, m2, m3, m4 = st.columns(4)

crr = (runs_scored / overs_decimal) if overs_decimal > 0 else 0.0

with m1:
    st.markdown(f'<div class="stat-card"><div class="stat-label">Current Run Rate (CRR)</div><div class="stat-value">{crr:.2f}</div></div>', unsafe_allow_html=True)

with m2:
    st.markdown(f'<div class="stat-card"><div class="stat-label">Balls Remaining</div><div class="stat-value">{balls_remaining}</div></div>', unsafe_allow_html=True)

if innings == 2:
    runs_needed = max(0, target_runs - runs_scored)
    overs_remaining = balls_remaining / 6.0 if balls_remaining > 0 else 0.001
    rrr = (runs_needed / overs_remaining) if overs_remaining > 0 else 0.0
    
    with m3:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Runs Needed</div><div class="stat-value">{runs_needed}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Req. Run Rate (RRR)</div><div class="stat-value">{rrr:.2f}</div></div>', unsafe_allow_html=True)
else:
    with m3:
        proj_linear = int(crr * 20) if crr > 0 else 0
        st.markdown(f'<div class="stat-card"><div class="stat-label">Linear Pace Proj.</div><div class="stat-value">{proj_linear}</div></div>', unsafe_allow_html=True)
    with m4:
        wickets_in_hand = 10 - wickets_lost
        st.markdown(f'<div class="stat-card"><div class="stat-label">Wickets in Hand</div><div class="stat-value">{wickets_in_hand}</div></div>', unsafe_allow_html=True)

st.write("")

# Predict Action
if st.button("🔮 Calculate Match Prediction", type="primary", use_container_width=True):
    try:
        batting_enc = le_batting.transform([batting_team])[0]
        bowling_enc = le_bowling.transform([bowling_team])[0]
    except ValueError:
        st.error("Selected teams are not recognized by the encoding models.")
        st.stop()

    if innings == 1:
        # First Innings Score Projection
        features_reg = np.array([[batting_enc, bowling_enc, runs_scored, wickets_lost, overs_decimal]])
        predicted_total_runs = reg.predict(features_reg)[0]
        
        st.markdown('<div class="prob-card">', unsafe_allow_html=True)
        st.markdown("### 📊 1st Innings Projected Final Score")
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.metric(label="Predicted Total Score", value=f"{int(round(predicted_total_runs))} Runs", delta=f"{predicted_total_runs - runs_scored:+.0f} runs to be scored")
        with col_res2:
            lower_bound = max(runs_scored, int(predicted_total_runs - 10))
            upper_bound = int(predicted_total_runs + 10)
            st.metric(label="Estimated Safe Target Range", value=f"{lower_bound} - {upper_bound} Runs")
            
        st.info(f"💡 Based on the current score of **{runs_scored}/{wickets_lost}** in **{overs_completed} overs**, **{batting_team}** is projected to score **{predicted_total_runs:.0f} runs** against **{bowling_team}**.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Second Innings Win Probability
        features_cls = np.array([[batting_enc, bowling_enc, runs_scored, wickets_lost, overs_decimal, innings, target_runs]])
        probabilities = clf.predict_proba(features_cls)[0]
        batting_win_prob = probabilities[1]
        bowling_win_prob = probabilities[0]

        st.markdown('<div class="prob-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Win Probability Breakdown")
        
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f'<div style="text-align: center;"><span class="team-badge" style="color: #38BDF8;">{batting_team} (Chasing)</span><h2>{batting_win_prob * 100:.1f}%</h2></div>', unsafe_allow_html=True)
            st.progress(float(batting_win_prob))
        with p_col2:
            st.markdown(f'<div style="text-align: center;"><span class="team-badge" style="color: #F87171;">{bowling_team} (Defending)</span><h2>{bowling_win_prob * 100:.1f}%</h2></div>', unsafe_allow_html=True)
            st.progress(float(bowling_win_prob))

        st.write("")
        if batting_win_prob >= 0.70:
            st.success(f"🎉 **{batting_team}** are dominating the chase with a strong **{batting_win_prob * 100:.1f}%** win probability!")
        elif batting_win_prob >= 0.45:
            st.info(f"🤝 The match is in a tight balance! Both teams have a strong chance to clinch victory.")
        else:
            st.error(f"⚡ **{bowling_team}** are commanding the game and holding **{batting_team}** under pressure.")
            
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.9rem; padding: 1rem 0;">
        🏏 <strong>IPL Winning Team & Score Predictor</strong> | Trained on Historical Ball-by-Ball Match Data<br>
        Created with ❤️ for IPL analytics by <strong>Shivraj</strong>.
    </div>
    """,
    unsafe_allow_html=True
)
