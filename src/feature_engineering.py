"""
Domain-specific cricket feature engineering pipeline.
Transforms raw ball-by-ball events into match state vectors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
from src.config import CLASSIFICATION_FEATURES, REGRESSION_FEATURES


def build_match_state_dataframe(df_merged: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Transforms granular ball-by-ball merged data into match state records
    with cumulative statistics, target scores, and win outcomes.
    """
    # Group at ball level
    over_stats = df_merged.groupby(
        ['id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball'],
        as_index=False
    ).agg(
        runs=('total_runs', 'sum'),
        wickets=('is_wicket', 'sum'),
        overs_decimal=('overs_decimal', 'max')
    )

    # Cumulative calculations per match and inning
    over_stats['cum_runs'] = over_stats.groupby(['id', 'inning'])['runs'].cumsum()
    over_stats['cum_wickets'] = over_stats.groupby(['id', 'inning'])['wickets'].cumsum()
    over_stats['overs_completed'] = over_stats['overs_decimal']

    # Compute target runs from 1st innings totals
    innings1_totals = df_merged[df_merged['inning'] == 1].groupby('id')['total_runs'].sum().reset_index()
    innings1_totals.rename(columns={'total_runs': 'target_runs'}, inplace=True)

    over_stats = over_stats.merge(innings1_totals, on='id', how='left')
    over_stats['target_runs'] = np.where(over_stats['inning'] == 2, over_stats['target_runs'].fillna(0), 0)

    # Derived domain metrics
    over_stats['crr'] = np.where(
        over_stats['overs_completed'] > 0,
        over_stats['cum_runs'] / over_stats['overs_completed'],
        0.0
    )
    
    over_stats['balls_bowled'] = (np.floor(over_stats['overs_completed']) * 6 + ((over_stats['overs_completed'] % 1) * 6)).round().astype(int)
    over_stats['balls_remaining'] = np.maximum(0, 120 - over_stats['balls_bowled'])
    over_stats['wickets_in_hand'] = np.maximum(0, 10 - over_stats['cum_wickets'])
    over_stats['runs_needed'] = np.where(
        over_stats['inning'] == 2,
        np.maximum(0, over_stats['target_runs'] - over_stats['cum_runs']),
        0
    )

    # Fit Team Label Encoders
    all_teams = pd.concat([over_stats['batting_team'], over_stats['bowling_team']]).dropna().unique()
    le_batting = LabelEncoder().fit(all_teams)
    le_bowling = LabelEncoder().fit(all_teams)

    over_stats['batting_team_enc'] = le_batting.transform(over_stats['batting_team'])
    over_stats['bowling_team_enc'] = le_bowling.transform(over_stats['bowling_team'])

    # Determine match winner and win status for 2nd innings
    if 'winner' in df_merged.columns:
        match_winners = df_merged[['id', 'winner']].drop_duplicates()
        over_stats = over_stats.merge(match_winners, on='id', how='left')
        over_stats['win_status'] = 0
        over_stats.loc[
            (over_stats['inning'] == 2) &
            (over_stats['batting_team'] == over_stats['winner']),
            'win_status'
        ] = 1

    return over_stats, le_batting, le_bowling


def encode_single_match_state(
    batting_team: str,
    bowling_team: str,
    runs_scored: int,
    wickets_lost: int,
    overs_completed: float,
    innings: int,
    target_runs: int,
    le_batting: LabelEncoder,
    le_bowling: LabelEncoder
) -> Dict[str, Any]:
    """Encodes and prepares feature vector with exact column names for real-time inference."""
    batting_enc = le_batting.transform([batting_team])[0]
    bowling_enc = le_bowling.transform([bowling_team])[0]

    overs_int = int(overs_completed)
    balls = int(round((overs_completed - overs_int) * 10))
    if balls > 5:
        balls = 5
    overs_decimal = overs_int + (balls / 6.0)

    crr = (runs_scored / overs_decimal) if overs_decimal > 0 else 0.0
    balls_bowled = int(overs_int * 6 + balls)
    balls_remaining = max(0, 120 - balls_bowled)
    wickets_in_hand = max(0, 10 - wickets_lost)

    # Use Pandas DataFrame with feature names to ensure compatibility with estimators
    features_cls = pd.DataFrame([{
        'batting_team_enc': batting_enc,
        'bowling_team_enc': bowling_enc,
        'cum_runs': runs_scored,
        'cum_wickets': wickets_lost,
        'overs_completed': overs_decimal,
        'inning': innings,
        'target_runs': target_runs
    }])[CLASSIFICATION_FEATURES]

    features_reg = pd.DataFrame([{
        'batting_team_enc': batting_enc,
        'bowling_team_enc': bowling_enc,
        'cum_runs': runs_scored,
        'cum_wickets': wickets_lost,
        'overs_completed': overs_decimal
    }])[REGRESSION_FEATURES]

    return {
        "features_cls": features_cls,
        "features_reg": features_reg,
        "overs_decimal": overs_decimal,
        "crr": crr,
        "balls_remaining": balls_remaining,
        "wickets_in_hand": wickets_in_hand,
        "batting_enc": batting_enc,
        "bowling_enc": bowling_enc
    }
