import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load datasets
deliveries = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv")

# Rename columns for consistency
deliveries.rename(columns={'match_id': 'id'}, inplace=True)

# Calculate overs_decimal: over starts at 0, so add 1 for better readability
deliveries['overs_decimal'] = deliveries['over'] + deliveries['ball'] / 6

# Merge deliveries with match info to get winner
df = deliveries.merge(matches[['id', 'winner']], on='id', how='left')

# Calculate cumulative runs and wickets
over_stats = df.groupby(['id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball']).agg(
    runs=('total_runs', 'sum'),
    wickets=('is_wicket', 'sum'),
    overs_decimal=('overs_decimal', 'max')
).reset_index()

over_stats['cum_runs'] = over_stats.groupby(['id', 'inning'])['runs'].cumsum()
over_stats['cum_wickets'] = over_stats.groupby(['id', 'inning'])['wickets'].cumsum()
over_stats['overs_completed'] = over_stats['overs_decimal']

# Calculate target runs: sum of 1st innings total
innings1_totals = df[df['inning'] == 1].groupby(['id'])['total_runs'].sum().reset_index()
innings1_totals.rename(columns={'total_runs': 'target_runs'}, inplace=True)

# Merge target runs into over_stats (only relevant for 2nd innings)
over_stats = over_stats.merge(innings1_totals, on='id', how='left')
over_stats['target_runs'] = np.where(over_stats['inning'] == 2, over_stats['target_runs'], 0)

# Label encode teams
all_teams = pd.concat([over_stats['batting_team'], over_stats['bowling_team']]).unique()
le_batting = LabelEncoder()
le_bowling = LabelEncoder()
le_batting.fit(all_teams)
le_bowling.fit(all_teams)

over_stats['batting_team_enc'] = le_batting.transform(over_stats['batting_team'])
over_stats['bowling_team_enc'] = le_bowling.transform(over_stats['bowling_team'])

# Add win_status column (only meaningful for 2nd innings)
over_stats = over_stats.merge(matches[['id', 'winner']], on='id', how='left')
over_stats['win_status'] = 0
over_stats.loc[
    (over_stats['inning'] == 2) &
    (over_stats['batting_team'] == over_stats['winner']),
    'win_status'
] = 1

# Classification Model (2nd innings)
features_cls = ['batting_team_enc', 'bowling_team_enc', 'cum_runs', 'cum_wickets', 'overs_completed', 'inning', 'target_runs']
X_cls = over_stats[features_cls]
y_cls = over_stats['win_status']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_cls, y_train_cls)
print(f"Classification model accuracy: {clf.score(X_test_cls, y_test_cls):.4f}")

# Regression Model (1st innings)
innings1_data = over_stats[over_stats['inning'] == 1]
innings1_totals = df[df['inning'] == 1].groupby(['id'])['total_runs'].sum().reset_index()
innings1_totals.rename(columns={'total_runs': 'final_total'}, inplace=True)
innings1_data = innings1_data.merge(innings1_totals, on='id', how='left')

X_reg = innings1_data[['batting_team_enc', 'bowling_team_enc', 'cum_runs', 'cum_wickets', 'overs_completed']]
y_reg = innings1_data['final_total']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)
print(f"Regression model R2 score: {reg.score(X_test_reg, y_test_reg):.4f}")

# Save models and encoders
joblib.dump(clf, 'ipl_win_predictor.pkl')
joblib.dump(reg, 'ipl_runs_predictor.pkl')
joblib.dump(le_batting, 'le_batting.pkl')
joblib.dump(le_bowling, 'le_bowling.pkl')

print("Models and encoders saved successfully.")
