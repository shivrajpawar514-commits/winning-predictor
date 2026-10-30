"""
Comprehensive failure handling tests:
1. Missing model files
2. Corrupt model files (initialization errors)
3. Unavailable datasets
4. Malformed dataset inputs
5. Malformed API requests (invalid overs, identical teams, unknown teams, bounds)
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from src.predictor import MatchPredictor
from src.data_loader import load_raw_data, clean_and_merge_data
from api import app, reset_predictor


@pytest.fixture
def client():
    reset_predictor()
    return TestClient(app)


# 1. Missing Model Files Tests
def test_predictor_missing_model_files(tmp_path):
    missing_classifier = tmp_path / "nonexistent_clf.pkl"
    with pytest.raises(FileNotFoundError, match="Missing required model artifact"):
        MatchPredictor(classifier_path=missing_classifier)


def test_health_check_when_models_missing(monkeypatch, tmp_path):
    # Point models to empty tmp_path
    missing_path = tmp_path / "missing_clf.pkl"
    monkeypatch.setattr("src.predictor.WIN_CLASSIFIER_PATH", missing_path)
    reset_predictor()
    
    test_client = TestClient(app)
    response = test_client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["models_loaded"] is False
    assert "error" in data


# 2. Corrupt Model Files (Initialization Errors)
def test_predictor_corrupt_model_files(tmp_path):
    corrupt_clf = tmp_path / "corrupt_clf.pkl"
    corrupt_clf.write_text("not a valid pickle file")
    
    # Create valid dummy files for the others so it reaches unpickling
    reg = tmp_path / "reg.pkl"
    le_b = tmp_path / "le_b.pkl"
    le_bowl = tmp_path / "le_bowl.pkl"
    reg.write_text("dummy")
    le_b.write_text("dummy")
    le_bowl.write_text("dummy")

    with pytest.raises(RuntimeError, match="Failed to initialize models"):
        MatchPredictor(
            classifier_path=corrupt_clf,
            regressor_path=reg,
            le_batting_path=le_b,
            le_bowling_path=le_bowl
        )


# 3. Unavailable Datasets Tests
def test_load_raw_data_unavailable_files(tmp_path):
    missing_deliv = tmp_path / "nonexistent_deliv.csv"
    missing_match = tmp_path / "nonexistent_match.csv"

    with pytest.raises(FileNotFoundError, match="Deliveries dataset not found"):
        load_raw_data(deliveries_path=missing_deliv, matches_path=missing_match)


# 4. Malformed Dataset Tests
def test_clean_and_merge_data_missing_columns():
    invalid_deliv = pd.DataFrame([{"foo": "bar"}])
    invalid_matches = pd.DataFrame([{"id": 1}])

    with pytest.raises(ValueError, match="missing required columns"):
        clean_and_merge_data(invalid_deliv, invalid_matches)


def test_clean_and_merge_data_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        clean_and_merge_data(pd.DataFrame(), pd.DataFrame())


# 5. Malformed API Requests Tests
def test_api_malformed_overs_invalid_ball_count(client):
    # 14.8 has 8 balls in over (> 5) -> invalid cricket notation
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "runs_scored": 120,
        "wickets_lost": 3,
        "overs_completed": 14.8,
        "target_runs": 185
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422


def test_api_malformed_overs_decimal_fraction(client):
    # 14.25 is not cricket notation O.B
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "runs_scored": 120,
        "wickets_lost": 3,
        "overs_completed": 14.25,
        "target_runs": 185
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422


def test_api_malformed_overs_negative(client):
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "runs_scored": 120,
        "wickets_lost": 3,
        "overs_completed": -1.0,
        "target_runs": 185
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422


def test_api_malformed_overs_exceeds_max(client):
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "runs_scored": 120,
        "wickets_lost": 3,
        "overs_completed": 20.1,
        "target_runs": 185
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422


def test_api_malformed_identical_teams(client):
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Chennai Super Kings",
        "runs_scored": 120,
        "wickets_lost": 3,
        "overs_completed": 14.2,
        "target_runs": 185
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422


def test_api_malformed_invalid_wickets_bound(client):
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "runs_scored": 120,
        "wickets_lost": 11,
        "overs_completed": 14.2,
        "target_runs": 185
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422


def test_api_malformed_invalid_target_runs(client):
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "runs_scored": 120,
        "wickets_lost": 3,
        "overs_completed": 14.2,
        "target_runs": 0
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422


def test_api_malformed_missing_required_fields(client):
    payload = {
        "batting_team": "Chennai Super Kings"
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 422
