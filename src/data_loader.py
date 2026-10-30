"""
Dataset ingestion, validation, and preprocessing for IPL match data.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from src.config import DELIVERIES_CSV, MATCHES_CSV


def load_raw_data(deliveries_path=DELIVERIES_CSV, matches_path=MATCHES_CSV) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw deliveries and matches CSV datasets with error checking."""
    deliv_p = Path(deliveries_path)
    match_p = Path(matches_path)

    if not deliv_p.exists():
        raise FileNotFoundError(
            f"Deliveries dataset not found at '{deliv_p}'. "
            f"Please ensure deliveries.csv is present in the project directory."
        )
    if not match_p.exists():
        raise FileNotFoundError(
            f"Matches dataset not found at '{match_p}'. "
            f"Please ensure matches.csv is present in the project directory."
        )

    try:
        deliveries = pd.read_csv(deliv_p)
    except Exception as e:
        raise ValueError(f"Failed to read deliveries CSV from '{deliv_p}': {e}") from e

    try:
        matches = pd.read_csv(match_p)
    except Exception as e:
        raise ValueError(f"Failed to read matches CSV from '{match_p}': {e}") from e

    return deliveries, matches


def clean_and_merge_data(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Cleans, formats, validates, and merges deliveries with match outcomes."""
    if deliveries is None or deliveries.empty:
        raise ValueError("Deliveries dataframe cannot be empty or None.")
    if matches is None or matches.empty:
        raise ValueError("Matches dataframe cannot be empty or None.")

    df_deliv = deliveries.copy()
    df_match = matches.copy()

    # Standardize match ID column name
    if 'match_id' in df_deliv.columns and 'id' not in df_deliv.columns:
        df_deliv.rename(columns={'match_id': 'id'}, inplace=True)

    required_deliv_cols = {'id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball', 'total_runs'}
    missing_deliv = required_deliv_cols - set(df_deliv.columns)
    if missing_deliv:
        raise ValueError(f"Deliveries dataset is missing required columns: {sorted(list(missing_deliv))}")

    if 'id' not in df_match.columns:
        raise ValueError("Matches dataset must contain an 'id' column.")

    if 'is_wicket' not in df_deliv.columns:
        df_deliv['is_wicket'] = 0

    # Compute continuous decimal overs: e.g. over 0, ball 1 -> 0.1667
    df_deliv['overs_decimal'] = df_deliv['over'] + (df_deliv['ball'] / 6.0)

    # Merge match winner, season, venue, and city information
    match_cols = ['id', 'winner', 'season', 'city', 'venue', 'toss_winner', 'toss_decision']
    available_cols = [c for c in match_cols if c in df_match.columns]
    
    df_merged = df_deliv.merge(df_match[available_cols], on='id', how='left')
    return df_merged

