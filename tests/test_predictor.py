"""
Unit tests for MatchPredictor inference engine.
"""

import pytest
from src.predictor import MatchPredictor


@pytest.fixture(scope="module")
def predictor():
    return MatchPredictor()


def test_predictor_initialization(predictor):
    assert len(predictor.teams) > 0
    assert "Chennai Super Kings" in predictor.teams
    assert "Mumbai Indians" in predictor.teams


def test_win_probability_prediction_bounds(predictor):
    res = predictor.predict_win_probability(
        batting_team="Chennai Super Kings",
        bowling_team="Mumbai Indians",
        runs_scored=120,
        wickets_lost=2,
        overs_completed=12.0,
        target_runs=170
    )

    assert 0.0 <= res["batting_win_probability"] <= 1.0
    assert 0.0 <= res["bowling_win_probability"] <= 1.0
    assert pytest.approx(res["batting_win_probability"] + res["bowling_win_probability"], 0.01) == 1.0
    assert res["runs_needed"] == 50
    assert res["balls_remaining"] == 48


def test_first_innings_score_bounds(predictor):
    res = predictor.predict_first_innings_score(
        batting_team="Chennai Super Kings",
        bowling_team="Mumbai Indians",
        runs_scored=90,
        wickets_lost=1,
        overs_completed=10.0
    )

    assert res["projected_total_runs"] >= 90
    assert res["score_lower_bound"] <= res["projected_total_runs"] <= res["score_upper_bound"]


def test_invalid_identical_teams_error(predictor):
    with pytest.raises(ValueError, match="identical"):
        predictor.predict_win_probability(
            batting_team="Chennai Super Kings",
            bowling_team="Chennai Super Kings",
            runs_scored=50,
            wickets_lost=1,
            overs_completed=6.0,
            target_runs=150
        )
