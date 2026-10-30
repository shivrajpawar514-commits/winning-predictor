"""
Domain-specific cricket feature engineering pipeline.
Transforms raw ball-by-ball events into match state vectors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
from src.config import CLASSIFICATION_FEATURES, REGRESSION_FEATURES, TEAM_ALIASES
from src.utils import validate_and_convert_overs, total_balls_to_decimal_overs


def build_match_state_dataframe(df_merged: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Transforms granular ball-by-ball merged data into match state records
    with cumulative statistics, target scores, and win outcomes.
    Preserves match 'id' and 'season' metadata to enable leak-free train/test splitting.
    """
    df = df_merged.copy()

    # Apply team aliases if present
    if 'batting_team' in df.columns:
        df['batting_team'] = df['batting_team'].replace(TEAM_ALIASES)
    if 'bowling_team' in df.columns:
        df['bowling_team'] = df['bowling_team'].replace(TEAM_ALIASES)
    if 'winner' in df.columns:
        df['winner'] = df['winner'].replace(TEAM_ALIASES)

    # Group at ball level
    over_stats = df.groupby(
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
    innings1_totals = df[df['inning'] == 1].groupby('id')['total_runs'].sum().reset_index()
    innings1_totals.rename(columns={'total_runs': 'target_runs'}, inplace=True)

    over_stats = over_stats.merge(innings1_totals, on='id', how='left')
    over_stats['target_runs'] = np.where(over_stats['inning'] == 2, over_stats['target_runs'].fillna(0), 0)

    # Derived domain metrics
    over_stats['crr'] = np.where(
        over_stats['overs_completed'] > 0,
        over_stats['cum_runs'] / over_stats['overs_completed'],
        0.0
    )
    
    # Calculate exact balls bowled and balls remaining
    # overs_decimal was computed as over + ball / 6.0 (or over * 6 + ball total balls)
    over_stats['balls_bowled'] = (over_stats['overs_completed'] * 6.0).round().astype(int)
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

    # Merge match winner, season, and win status for 2nd innings
    match_meta_cols = ['id']
    if 'winner' in df.columns:
        match_meta_cols.append('winner')
    if 'season' in df.columns:
        match_meta_cols.append('season')

    match_meta = df[match_meta_cols].drop_duplicates()
    over_stats = over_stats.merge(match_meta, on='id', how='left')
    
    over_stats['win_status'] = 0
    if 'winner' in over_stats.columns:
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
    # Apply aliases if provided
    batting_clean = TEAM_ALIASES.get(batting_team, batting_team)
    bowling_clean = TEAM_ALIASES.get(bowling_team, bowling_team)

    batting_enc = le_batting.transform([batting_clean])[0]
    bowling_enc = le_bowling.transform([bowling_clean])[0]

    # Convert cricket overs (e.g. 14.2 = 14 overs 2 balls) to total balls and continuous decimal overs
    total_balls = validate_and_convert_overs(overs_completed)
    overs_decimal = total_balls_to_decimal_overs(total_balls)

    crr = (runs_scored / overs_decimal) if overs_decimal > 0 else 0.0
    balls_bowled = total_balls
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
        "total_balls": total_balls,
        "crr": crr,
        "balls_remaining": balls_remaining,
        "wickets_in_hand": wickets_in_hand,
        "batting_enc": batting_enc,
        "bowling_enc": bowling_enc
    }

