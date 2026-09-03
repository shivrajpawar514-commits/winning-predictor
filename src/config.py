"""
Central configuration for IPL Match Predictor ML Pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR

DELIVERIES_CSV = DATA_DIR / "deliveries.csv"
MATCHES_CSV = DATA_DIR / "matches.csv"

WIN_CLASSIFIER_PATH = MODELS_DIR / "ipl_win_predictor.pkl"
RUNS_REGRESSOR_PATH = MODELS_DIR / "ipl_runs_predictor.pkl"
LE_BATTING_PATH = MODELS_DIR / "le_batting.pkl"
LE_BOWLING_PATH = MODELS_DIR / "le_bowling.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Feature definitions
CLASSIFICATION_FEATURES = [
    'batting_team_enc',
    'bowling_team_enc',
    'cum_runs',
    'cum_wickets',
    'overs_completed',
    'inning',
    'target_runs'
]

REGRESSION_FEATURES = [
    'batting_team_enc',
    'bowling_team_enc',
    'cum_runs',
    'cum_wickets',
    'overs_completed'
]

# Team standard aliases mapping
TEAM_ALIASES = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}
