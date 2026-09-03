"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "IPL Match Predictor API"
    assert data["version"] == "2.0.0"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_list_teams_endpoint():
    response = client.get("/teams")
    assert response.status_code == 200
    teams = response.json()
    assert isinstance(teams, list)
    assert len(teams) > 0


def test_predict_win_probability_endpoint():
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "runs_scored": 110,
        "wickets_lost": 2,
        "overs_completed": 12.0,
        "target_runs": 165
    }
    response = client.post("/predict/win-probability", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "batting_win_probability" in data
    assert 0.0 <= data["batting_win_probability"] <= 1.0
    assert data["runs_needed"] == 55


def test_predict_first_innings_score_endpoint():
    payload = {
        "batting_team": "Gujarat Titans",
        "bowling_team": "Rajasthan Royals",
        "runs_scored": 80,
        "wickets_lost": 1,
        "overs_completed": 9.0
    }
    response = client.post("/predict/first-innings-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "projected_total_runs" in data
    assert data["projected_total_runs"] >= 80


def test_simulate_scenario_endpoint():
    payload = {
        "batting_team": "Kolkata Knight Riders",
        "bowling_team": "Sunrisers Hyderabad",
        "current_runs": 95,
        "current_wickets": 2,
        "current_over": 10.0,
        "target_runs": 175,
        "run_rates": [8.0, 10.0],
        "simulate_wickets_lost": 1
    }
    response = client.post("/simulate/scenario", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "trajectories" in data
    assert len(data["trajectories"]) == 2
