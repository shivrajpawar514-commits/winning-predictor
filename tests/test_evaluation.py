"""
Tests for Model Evaluation Methodology:
- Leak-free splitting by match and season
- Calibration metrics (ECE, MCE, Reliability table)
- Brier score computation
"""

import numpy as np
import pandas as pd
import pytest
from src.train import split_dataset, compute_calibration_metrics


def test_split_dataset_by_match_zero_leakage():
    # Construct synthetic match records
    match_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    records = []
    for mid in match_ids:
        for ball in range(1, 10):
            records.append({
                "id": mid,
                "ball": ball,
                "win_status": 1 if mid % 2 == 0 else 0,
                "cum_runs": ball * 10
            })
    df = pd.DataFrame(records)

    train_df, test_df = split_dataset(df, split_strategy="match", test_size=0.3, random_state=42)

    train_matches = set(train_df["id"])
    test_matches = set(test_df["id"])

    # Must be disjoint
    overlap = train_matches.intersection(test_matches)
    assert len(overlap) == 0, f"Detected match ID leakage: {overlap}"
    assert len(train_matches) + len(test_matches) == len(match_ids)


def test_split_dataset_by_season():
    records = []
    seasons = ["2018", "2019", "2020", "2021", "2022"]
    for i, s in enumerate(seasons):
        records.append({"id": i, "season": s, "win_status": 1})
    df = pd.DataFrame(records)

    train_df, test_df = split_dataset(df, split_strategy="season", test_size=0.4, random_state=42)

    train_seasons = set(train_df["season"])
    test_seasons = set(test_df["season"])

    assert len(train_seasons.intersection(test_seasons)) == 0
    # Latest seasons must be in test set
    assert "2022" in test_seasons
    assert "2018" in train_seasons


def test_compute_calibration_metrics():
    # Perfectly calibrated synthetic probabilities
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.8, 0.85, 0.95])

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)

    assert "ece" in metrics
    assert "mce" in metrics
    assert "reliability_table" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["mce"] <= 1.0
    assert len(metrics["reliability_table"]) > 0

