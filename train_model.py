"""
IPL Match Prediction Pipeline
Trains Random Forest Classifier for 2nd Innings Win Probability
and Random Forest Regressor for 1st Innings Score Projection
using leak-free match-level group splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from src.config import TEAM_ALIASES


def prepare_dataset(deliveries_path="deliveries.csv", matches_path="matches.csv"):
    """Loads and transforms ball-by-ball and match dataset into feature vectors."""
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError(f"Deliveries dataset not found at '{deliveries_path}'.")
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Matches dataset not found at '{matches_path}'.")

    print("📥 Loading datasets...")
    deliveries = pd.read_csv(deliveries_path)
    matches = pd.read_csv(matches_path)

    # Rename columns for consistency
    if 'match_id' in deliveries.columns:
        deliveries.rename(columns={'match_id': 'id'}, inplace=True)

    # Harmonize team names
    deliveries['batting_team'] = deliveries['batting_team'].replace(TEAM_ALIASES)
    deliveries['bowling_team'] = deliveries['bowling_team'].replace(TEAM_ALIASES)
    matches['winner'] = matches['winner'].replace(TEAM_ALIASES)

    # Calculate overs_decimal
    deliveries['overs_decimal'] = deliveries['over'] + deliveries['ball'] / 6.0

    # Merge deliveries with match info to get winner and season
    match_cols = ['id', 'winner']
    if 'season' in matches.columns:
        match_cols.append('season')
    df = deliveries.merge(matches[match_cols], on='id', how='left')

    print("⚙️ Computing cumulative match stats...")
    # Calculate cumulative runs and wickets
    over_stats = df.groupby(['id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball']).agg(
        runs=('total_runs', 'sum'),
        wickets=('is_wicket', 'sum'),
        overs_decimal=('overs_decimal', 'max')
    ).reset_index()

    over_stats['cum_runs'] = over_stats.groupby(['id', 'inning'])['runs'].cumsum()
    over_stats['cum_wickets'] = over_stats.groupby(['id', 'inning'])['wickets'].cumsum()
    over_stats['overs_completed'] = over_stats['overs_decimal']

    # Calculate target runs from 1st innings total
    innings1_totals = df[df['inning'] == 1].groupby(['id'])['total_runs'].sum().reset_index()
    innings1_totals.rename(columns={'total_runs': 'target_runs'}, inplace=True)

    # Merge target runs into over_stats (only relevant for 2nd innings)
    over_stats = over_stats.merge(innings1_totals, on='id', how='left')
    over_stats['target_runs'] = np.where(over_stats['inning'] == 2, over_stats['target_runs'].fillna(0), 0)

    # Label encode teams
    all_teams = pd.concat([over_stats['batting_team'], over_stats['bowling_team']]).dropna().unique()
    le_batting = LabelEncoder()
    le_bowling = LabelEncoder()
    le_batting.fit(all_teams)
    le_bowling.fit(all_teams)

    over_stats['batting_team_enc'] = le_batting.transform(over_stats['batting_team'])
    over_stats['bowling_team_enc'] = le_bowling.transform(over_stats['bowling_team'])

    # Add win_status column for 2nd innings
    match_winners = matches[['id', 'winner']].drop_duplicates()
    over_stats = over_stats.merge(match_winners, on='id', how='left')
    over_stats['win_status'] = 0
    over_stats.loc[
        (over_stats['inning'] == 2) &
        (over_stats['batting_team'] == over_stats['winner']),
        'win_status'
    ] = 1

    return df, over_stats, le_batting, le_bowling


def train_classifier(over_stats):
    """Trains Random Forest Classifier for 2nd Innings Win Probability with leak-free match splitting."""
    print("\n🌲 Training Classification Model (Win Probability) with Leak-Free Match Split...")
    second_innings = over_stats[over_stats['inning'] == 2].copy()
    features_cls = ['batting_team_enc', 'bowling_team_enc', 'cum_runs', 'cum_wickets', 'overs_completed', 'inning', 'target_runs']

    # Leak-free split by match id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(second_innings, groups=second_innings['id']))

    train_data = second_innings.iloc[train_idx]
    test_data = second_innings.iloc[test_idx]

    # Verify leak-free condition
    assert len(set(train_data['id']).intersection(set(test_data['id']))) == 0, "Data leakage detected across match IDs!"

    X_train = train_data[features_cls]
    y_train = train_data['win_status'].values
    X_test = test_data[features_cls]
    y_test = test_data['win_status'].values

    clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print(f"✅ Classification Accuracy: {acc * 100:.2f}% | ROC-AUC: {auc:.4f} | Brier Score: {brier:.4f}")
    return clf


def train_regressor(df, over_stats):
    """Trains Random Forest Regressor for 1st Innings Total Score with leak-free match splitting."""
    print("\n📈 Training Regression Model (1st Innings Projected Score) with Leak-Free Match Split...")
    innings1_data = over_stats[over_stats['inning'] == 1].copy()
    innings1_totals = df[df['inning'] == 1].groupby(['id'])['total_runs'].sum().reset_index()
    innings1_totals.rename(columns={'total_runs': 'final_total'}, inplace=True)
    innings1_data = innings1_data.merge(innings1_totals, on='id', how='left')

    features_reg = ['batting_team_enc', 'bowling_team_enc', 'cum_runs', 'cum_wickets', 'overs_completed']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(innings1_data, groups=innings1_data['id']))

    train_data = innings1_data.iloc[train_idx]
    test_data = innings1_data.iloc[test_idx]

    assert len(set(train_data['id']).intersection(set(test_data['id']))) == 0, "Data leakage detected across match IDs!"

    X_train = train_data[features_reg]
    y_train = train_data['final_total'].values
    X_test = test_data[features_reg]
    y_test = test_data['final_total'].values

    reg = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)

    y_pred_reg = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
    cov10 = np.mean(np.abs(y_test - y_pred_reg) <= 10)
    print(f"✅ Regression R² Score: {r2:.4f} (RMSE: {rmse:.2f} runs | Coverage ±10 runs: {cov10 * 100:.1f}%)")
    return reg


def main():
    df, over_stats, le_batting, le_bowling = prepare_dataset()
    clf = train_classifier(over_stats)
    reg = train_regressor(df, over_stats)

    print("\n💾 Saving trained models and encoders...")
    joblib.dump(clf, 'ipl_win_predictor.pkl')
    joblib.dump(reg, 'ipl_runs_predictor.pkl')
    joblib.dump(le_batting, 'le_batting.pkl')
    joblib.dump(le_bowling, 'le_bowling.pkl')
    print("🎉 All models and encoders saved successfully!")


if __name__ == "__main__":
    main()

