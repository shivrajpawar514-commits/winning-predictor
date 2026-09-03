"""
Unit tests for feature engineering computations.
"""

import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.feature_engineering import encode_single_match_state


@pytest.fixture
def mock_encoders():
    teams = ["Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders"]
    le_b = LabelEncoder().fit(teams)
    le_bowl = LabelEncoder().fit(teams)
    return le_b, le_bowl


def test_encode_single_match_state_shapes(mock_encoders):
    le_b, le_bowl = mock_encoders
    res = encode_single_match_state(
        batting_team="Chennai Super Kings",
        bowling_team="Mumbai Indians",
        runs_scored=100,
        wickets_lost=2,
        overs_completed=10.0,
        innings=2,
        target_runs=180,
        le_batting=le_b,
        le_bowling=le_bowl
    )

    assert res["features_cls"].shape == (1, 7)
    assert res["features_reg"].shape == (1, 5)
    assert res["balls_remaining"] == 60
    assert res["wickets_in_hand"] == 8
    assert res["crr"] == 10.0


def test_overs_decimal_conversion(mock_encoders):
    le_b, le_bowl = mock_encoders
    # 7.3 overs = 7 + 3/6 = 7.5 decimal overs
    res = encode_single_match_state(
        batting_team="Chennai Super Kings",
        bowling_team="Mumbai Indians",
        runs_scored=75,
        wickets_lost=1,
        overs_completed=7.3,
        innings=1,
        target_runs=0,
        le_batting=le_b,
        le_bowling=le_bowl
    )

    assert pytest.approx(res["overs_decimal"], 0.01) == 7.5
    assert pytest.approx(res["crr"], 0.01) == 10.0
