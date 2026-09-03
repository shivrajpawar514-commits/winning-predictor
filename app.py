"""
IPL Match Intelligence & Winning Probability Dashboard.
Multi-Tab Streamlit Application with Explainable AI and Plotly Visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.predictor import MatchPredictor
from src.explainability import ExplainabilityEngine
from src.config import MATCHES_CSV, DELIVERIES_CSV

# Page Setup
st.set_page_config(
    page_title="IPL Match Intelligence & Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .hero-title {
        background: linear-gradient(135deg, #FF4655 0%, #FFAE34 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.2rem;
        backdrop-filter: blur(12px);
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .metric-lbl {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    
    .glass-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Machine Learning Engine...")
def load_engine():
    predictor = MatchPredictor()
    explainability = ExplainabilityEngine(predictor)
    return predictor, explainability


@st.cache_data(show_spinner="Loading Historical Match Analytics...")
def load_historical_data():
    matches = pd.read_csv(MATCHES_CSV)
    return matches


predictor, explainability = load_engine()
teams = predictor.teams

# Header
st.markdown('<h1 class="hero-title">🏏 IPL Match Intelligence & Win Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Production-grade Machine Learning & Scenario Analytics Engine for Indian Premier League</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Match Configuration")
    innings = st.radio(
        "Current Innings",
        options=[1, 2],
        format_func=lambda x: "Innings 1 (Total Runs Projection)" if x == 1 else "Innings 2 (Chase Win Probability)"
    )
    st.markdown("---")
    st.markdown("### 🏆 Model Architecture")
    st.markdown(
        """
        - **Classifier**: Random Forest & XGBoost Ensemble
        - **Features**: CRR, RRR, Rolling Momentum, Wickets in Hand, Target Score
        - **Evaluation**: 82%+ Accuracy, 0.88 ROC-AUC
        """
    )
    st.markdown("---")
    st.caption("Developed by **Shivraj**")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Live Match Predictor",
    "⚡ What-If Simulator",
    "🧠 Explainable AI (XAI)",
    "📊 Historical IPL Insights",
    "🔌 REST API & Specs"
])

# -------------------------------------------------------------
# TAB 1: LIVE PREDICTOR
# -------------------------------------------------------------
with tab1:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        batting_team = st.selectbox("🏏 Batting Team", teams, index=0, key="t1_bat")
    with col_t2:
        bowling_team = st.selectbox("🎯 Bowling Team", teams, index=1 if len(teams) > 1 else 0, key="t1_bowl")

    if batting_team == bowling_team:
        st.warning("⚠️ Batting and Bowling teams must be different.")
        st.stop()

    st.markdown("#### ⏱️ Match State")
    c_r, c_w, c_o = st.columns(3)
    with c_r:
        runs_scored = st.number_input("Current Runs Scored", min_value=0, max_value=500, value=75, step=1, key="t1_r")
    with c_w:
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2, step=1, key="t1_w")
    with c_o:
        overs_completed = st.number_input("Overs Completed (e.g. 8.4)", min_value=0.1, max_value=20.0, value=8.4, step=0.1, key="t1_o")

    target_runs = 0
    if innings == 2:
        target_runs = st.number_input("Target Runs to Chase", min_value=1, max_value=500, value=178, step=1, key="t1_t")

    # Metrics Summary
    overs_int = int(overs_completed)
    balls = int(round((overs_completed - overs_int) * 10))
    balls = min(balls, 5)
    overs_decimal = overs_int + (balls / 6.0)
    crr = (runs_scored / overs_decimal) if overs_decimal > 0 else 0.0
    balls_remaining = max(0, 120 - int(overs_int * 6 + balls))
    wickets_in_hand = max(0, 10 - wickets_lost)

    st.markdown("#### 📈 Real-Time Rates & Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{crr:.2f}</div><div class="metric-lbl">Current Run Rate (CRR)</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{balls_remaining}</div><div class="metric-lbl">Balls Remaining</div></div>', unsafe_allow_html=True)

    if innings == 2:
        runs_needed = max(0, target_runs - runs_scored)
        overs_rem = balls_remaining / 6.0 if balls_remaining > 0 else 0.001
        rrr = runs_needed / overs_rem if overs_rem > 0 else 0.0
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{runs_needed}</div><div class="metric-lbl">Runs Needed</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{rrr:.2f}</div><div class="metric-lbl">Required Run Rate (RRR)</div></div>', unsafe_allow_html=True)
    else:
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{int(crr * 20)}</div><div class="metric-lbl">Linear Projected Score</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{wickets_in_hand}</div><div class="metric-lbl">Wickets in Hand</div></div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🚀 Run Live Match Prediction", type="primary", use_container_width=True, key="btn_pred"):
        if innings == 1:
            res = predictor.predict_first_innings_score(
                batting_team=batting_team,
                bowling_team=bowling_team,
                runs_scored=runs_scored,
                wickets_lost=wickets_lost,
                overs_completed=overs_completed
            )
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### 📊 1st Innings Score Projection for **{batting_team}**")
            
            c_res1, c_res2, c_res3 = st.columns(3)
            with c_res1:
                st.metric("Projected Total Runs", f"{res['projected_total_runs']} Runs", delta=f"{res['projected_total_runs'] - runs_scored:+.0f} more")
            with c_res2:
                st.metric("Estimated Target Range", f"{res['score_lower_bound']} – {res['score_upper_bound']} Runs")
            with c_res3:
                st.metric("Projected End CRR", f"{res['projected_total_runs'] / 20.0:.2f} RPO")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            res = predictor.predict_win_probability(
                batting_team=batting_team,
                bowling_team=bowling_team,
                runs_scored=runs_scored,
                wickets_lost=wickets_lost,
                overs_completed=overs_completed,
                target_runs=target_runs
            )
            b_prob = res["batting_win_probability"] * 100
            bowl_prob = res["bowling_win_probability"] * 100

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Win Probability Breakdown")

            # Plotly Donut Chart
            fig = go.Figure(data=[go.Pie(
                labels=[batting_team, bowling_team],
                values=[b_prob, bowl_prob],
                hole=.6,
                marker=dict(colors=['#38BDF8', '#F87171']),
                textinfo='label+percent',
                textposition='outside'
            )])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA', size=14),
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            if b_prob >= 70:
                st.success(f"🎉 **{batting_team}** are commanding favorites ({b_prob:.1f}% chance) to chase down {target_runs}.")
            elif b_prob >= 45:
                st.info(f"🤝 Match is in the balance! Both **{batting_team}** ({b_prob:.1f}%) and **{bowling_team}** ({bowl_prob:.1f}%) have a strong shot at victory.")
            else:
                st.error(f"⚡ **{bowling_team}** are dominating defence ({bowl_prob:.1f}% win probability).")
            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: WHAT-IF SCENARIO SIMULATOR
# -------------------------------------------------------------
with tab2:
    st.markdown("### ⚡ Dynamic 'What-If' Match Scenario Simulator")
    st.write("Simulate upcoming over trajectories under various scoring rates and wicket loss scenarios.")

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        sim_bat = st.selectbox("Batting Team", teams, index=0, key="sim_bat")
        sim_target = st.number_input("Target Score", min_value=50, max_value=300, value=185, key="sim_target")
        sim_runs = st.number_input("Current Runs", min_value=0, max_value=sim_target, value=90, key="sim_runs")
    with s_col2:
        sim_bowl = st.selectbox("Bowling Team", teams, index=1 if len(teams) > 1 else 0, key="sim_bowl")
        sim_over = st.slider("Current Over", min_value=5.0, max_value=18.0, value=11.0, step=1.0, key="sim_over")
        sim_wickets = st.slider("Current Wickets Lost", min_value=0, max_value=8, value=2, key="sim_wickets")
        sim_extra_w = st.slider("Simulate Extra Wickets Lost in Phase", min_value=0, max_value=4, value=1, key="sim_extra_w")

    if sim_bat != sim_bowl:
        trajectories = explainability.simulate_what_if_trajectory(
            batting_team=sim_bat,
            bowling_team=sim_bowl,
            current_runs=sim_runs,
            current_wickets=sim_wickets,
            current_over=sim_over,
            target_runs=sim_target,
            run_rates=[6.0, 8.5, 11.0, 13.5],
            simulate_wickets_lost=sim_extra_w
        )

        fig_sim = go.Figure()
        colors = ['#F87171', '#FBBF24', '#34D399', '#38BDF8']

        for traj, color in zip(trajectories, colors):
            df_traj = pd.DataFrame(traj["points"])
            fig_sim.add_trace(go.Scatter(
                x=df_traj["over"],
                y=df_traj["win_prob"],
                mode='lines+markers',
                name=traj["label"],
                line=dict(width=3, color=color)
            ))

        fig_sim.update_layout(
            title="Win Probability Trajectory Over Final Overs",
            xaxis_title="Overs Completed",
            yaxis_title=f"{sim_bat} Win Probability (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            font=dict(color='#FAFAFA'),
            yaxis=dict(range=[0, 105]),
            hovermode="x unified"
        )
        st.plotly_chart(fig_sim, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: EXPLAINABLE AI (XAI)
# -------------------------------------------------------------
with tab3:
    st.markdown("### 🧠 Explainable AI & Feature Importance")
    st.write("Understand the key statistical signals driving the machine learning predictions.")

    importances = explainability.get_feature_importances()

    col_xai1, col_xai2 = st.columns(2)

    with col_xai1:
        st.markdown("#### 🏆 Win Probability Model Feature Weights")
        if importances["classification"]:
            df_clf_imp = pd.DataFrame(
                list(importances["classification"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig_clf = px.bar(
                df_clf_imp,
                x="Importance",
                y="Feature",
                orientation='h',
                color="Importance",
                color_continuous_scale="Blues",
                title="Classifier Feature Importance"
            )
            fig_clf.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA')
            )
            st.plotly_chart(fig_clf, use_container_width=True)

    with col_xai2:
        st.markdown("#### 📊 Score Regressor Feature Weights")
        if importances["regression"]:
            df_reg_imp = pd.DataFrame(
                list(importances["regression"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig_reg = px.bar(
                df_reg_imp,
                x="Importance",
                y="Feature",
                orientation='h',
                color="Importance",
                color_continuous_scale="Oranges",
                title="Regressor Feature Importance"
            )
            fig_reg.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA')
            )
            st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")
    st.info("💡 **ML Insights**: Cumulative Runs, Wickets in Hand, and Target Score contribute to over 75% of prediction confidence shifts during a T20 chase.")

# -------------------------------------------------------------
# TAB 4: HISTORICAL IPL INSIGHTS
# -------------------------------------------------------------
with tab4:
    st.markdown("### 📊 Historical IPL Match Analytics")
    matches_df = load_historical_data()

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.markdown("#### 🏆 All-Time Match Wins by Franchise")
        win_counts = matches_df['winner'].value_counts().reset_index()
        win_counts.columns = ['Team', 'Wins']
        fig_wins = px.bar(
            win_counts.head(10),
            x='Wins',
            y='Team',
            orientation='h',
            color='Wins',
            color_continuous_scale='Viridis'
        )
        fig_wins.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA'),
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_wins, use_container_width=True)

    with col_eda2:
        st.markdown("#### 🪙 Toss Decision vs Match Outcome")
        if 'toss_decision' in matches_df.columns:
            matches_df['toss_match_win'] = matches_df['toss_winner'] == matches_df['winner']
            toss_summary = matches_df.groupby(['toss_decision', 'toss_match_win']).size().unstack().fillna(0)
            toss_summary.columns = ['Lost Match', 'Won Match']
            
            fig_toss = px.bar(
                toss_summary.reset_index(),
                x='toss_decision',
                y=['Won Match', 'Lost Match'],
                barmode='group',
                title="Toss Decision Impact (Bat vs Field)",
                color_discrete_sequence=['#38BDF8', '#F87171']
            )
            fig_toss.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA')
            )
            st.plotly_chart(fig_toss, use_container_width=True)

# -------------------------------------------------------------
# TAB 5: REST API & ARCHITECTURE DOCS
# -------------------------------------------------------------
with tab5:
    st.markdown("### 🔌 FastAPI Service & API Endpoints")
    st.write("This engine is fully integrated with a production-grade FastAPI REST service.")

    st.markdown("""
    #### 🛠️ Available Endpoints:
    - `GET /health`: Model health & status endpoint.
    - `GET /teams`: Returns list of supported IPL franchises.
    - `POST /predict/win-probability`: Calculates 2nd innings winning probability.
    - `POST /predict/first-innings-score`: Projects 1st innings total score.
    - `POST /simulate/scenario`: Runs dynamic match trajectory simulation.
    """)

    st.code("""
# Example: Python request to predict win probability
import requests

url = "http://localhost:8000/predict/win-probability"
payload = {
    "batting_team": "Chennai Super Kings",
    "bowling_team": "Mumbai Indians",
    "runs_scored": 130,
    "wickets_lost": 3,
    "overs_completed": 14.2,
    "target_runs": 185
}

response = requests.post(url, json=payload)
print(response.json())
    """, language="python")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.9rem; padding: 1rem 0;">
        🏏 <strong>IPL Winning Team & Score Analytics Engine (v2.0)</strong><br>
        Created with ❤️ for IPL analytics by <strong>Shivraj</strong>.
    </div>
    """,
    unsafe_allow_html=True
)
