"""
Dataset ingestion, validation, and preprocessing for IPL match data.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.config import DELIVERIES_CSV, MATCHES_CSV


def load_raw_data(deliveries_path=DELIVERIES_CSV, matches_path=MATCHES_CSV) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw deliveries and matches CSV datasets."""
    if not deliveries_path.exists():
        raise FileNotFoundError(f"Deliveries dataset not found at {deliveries_path}")
    if not matches_path.exists():
        raise FileNotFoundError(f"Matches dataset not found at {matches_path}")

    deliveries = pd.read_csv(deliveries_path)
    matches = pd.read_csv(matches_path)
    return deliveries, matches


def clean_and_merge_data(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Cleans, formats, and merges deliveries with match outcomes."""
    df_deliv = deliveries.copy()
    df_match = matches.copy()

    # Standardize match ID column name
    if 'match_id' in df_deliv.columns:
        df_deliv.rename(columns={'match_id': 'id'}, inplace=True)

    # Compute continuous decimal overs: e.g. over 0, ball 1 -> 0.1667
    df_deliv['overs_decimal'] = df_deliv['over'] + (df_deliv['ball'] / 6.0)

    # Merge match winner, venue, and city information
    match_cols = ['id', 'winner', 'city', 'venue', 'toss_winner', 'toss_decision']
    available_cols = [c for c in match_cols if c in df_match.columns]
    
    df_merged = df_deliv.merge(df_match[available_cols], on='id', how='left')
    return df_merged
