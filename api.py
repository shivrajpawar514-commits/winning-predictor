"""
Enterprise FastAPI REST Service for IPL Match Predictions & Analytics.
Provides OpenAPI / Swagger documentation, validation, and health checks.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.predictor import MatchPredictor
from src.explainability import ExplainabilityEngine
from src.utils import (
    estimate_dls_par_score,
    get_match_momentum_indicator,
    calculate_strike_rate,
    calculate_economy_rate
)

app = FastAPI(
    title="🏏 IPL Match Predictor API",
    description="Enterprise Machine Learning REST API for real-time IPL winning probabilities, projected scores, DLS calculations, and match scenario simulations.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of predictor
_predictor: Optional[MatchPredictor] = None
_explainability: Optional[ExplainabilityEngine] = None


def get_predictor() -> MatchPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MatchPredictor()
    return _predictor


def get_explainability() -> ExplainabilityEngine:
    global _explainability
    if _explainability is None:
        _explainability = ExplainabilityEngine(get_predictor())
    return _explainability


# --- Request & Response Schemas ---

class WinProbabilityRequest(BaseModel):
    batting_team: str = Field(..., description="Name of the team currently batting/chasing", json_schema_extra={"example": "Chennai Super Kings"})
    bowling_team: str = Field(..., description="Name of the defending bowling team", json_schema_extra={"example": "Mumbai Indians"})
    runs_scored: int = Field(..., ge=0, le=500, description="Total runs scored so far in 2nd innings", json_schema_extra={"example": 120})
    wickets_lost: int = Field(..., ge=0, le=10, description="Wickets lost by batting team", json_schema_extra={"example": 3})
    overs_completed: float = Field(..., ge=0.0, le=20.0, description="Overs completed (e.g. 14.2 for 14 overs 2 balls)", json_schema_extra={"example": 14.2})
    target_runs: int = Field(..., ge=1, le=500, description="Target score required to win", json_schema_extra={"example": 185})


class WinProbabilityResponse(BaseModel):
    batting_team: str
    bowling_team: str
    batting_win_probability: float = Field(..., description="Chasing team win probability [0.0 - 1.0]")
    bowling_win_probability: float = Field(..., description="Defending team win probability [0.0 - 1.0]")
    current_run_rate: float
    required_run_rate: float
    runs_needed: int
    balls_remaining: int
    wickets_in_hand: int
    momentum_status: Optional[str] = None


class FirstInningsScoreRequest(BaseModel):
    batting_team: str = Field(..., description="Team batting first", json_schema_extra={"example": "Royal Challengers Bengaluru"})
    bowling_team: str = Field(..., description="Team bowling first", json_schema_extra={"example": "Kolkata Knight Riders"})
    runs_scored: int = Field(..., ge=0, le=500, description="Runs scored so far", json_schema_extra={"example": 85})
    wickets_lost: int = Field(..., ge=0, le=10, description="Wickets lost so far", json_schema_extra={"example": 2})
    overs_completed: float = Field(..., ge=0.1, le=20.0, description="Overs completed", json_schema_extra={"example": 10.0})


class FirstInningsScoreResponse(BaseModel):
    batting_team: str
    bowling_team: str
    projected_total_runs: int
    score_lower_bound: int
    score_upper_bound: int
    current_run_rate: float
    balls_remaining: int
    wickets_in_hand: int


class ScenarioSimulationRequest(BaseModel):
    batting_team: str = Field(..., json_schema_extra={"example": "Gujarat Titans"})
    bowling_team: str = Field(..., json_schema_extra={"example": "Rajasthan Royals"})
    current_runs: int = Field(..., ge=0, json_schema_extra={"example": 100})
    current_wickets: int = Field(..., ge=0, le=10, json_schema_extra={"example": 3})
    current_over: float = Field(..., ge=1.0, le=19.0, json_schema_extra={"example": 12.0})
    target_runs: int = Field(..., ge=1, json_schema_extra={"example": 180})
    run_rates: List[float] = Field(default=[6.0, 8.0, 10.0, 12.0])
    simulate_wickets_lost: int = Field(default=0, ge=0, le=5)


class DlsCalculationRequest(BaseModel):
    target_runs: int = Field(..., ge=1, le=500, description="Target score set by 1st innings", json_schema_extra={"example": 185})
    current_over: float = Field(..., ge=0.0, le=20.0, description="Overs completed in chase", json_schema_extra={"example": 11.4})
    wickets_lost: int = Field(..., ge=0, le=10, description="Wickets lost by chasing team", json_schema_extra={"example": 2})
    current_runs: int = Field(..., ge=0, le=500, description="Current runs scored by chasing team", json_schema_extra={"example": 105})


class DlsCalculationResponse(BaseModel):
    target_runs: int
    current_over: float
    wickets_lost: int
    current_runs: int
    dls_par_score: int
    par_differential: int
    is_ahead_of_par: bool


# --- Endpoints ---

@app.get("/", tags=["General"])
def root():
    return {
        "name": "IPL Match Predictor API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
def health_check():
    predictor = get_predictor()
    return {
        "status": "healthy",
        "models_loaded": True,
        "available_teams_count": len(predictor.teams)
    }


@app.get("/teams", response_model=List[str], tags=["Metadata"])
def list_teams():
    """Returns list of supported IPL franchises."""
    return get_predictor().teams


@app.post("/predict/win-probability", response_model=WinProbabilityResponse, tags=["Predictions"])
def predict_win_probability(req: WinProbabilityRequest):
    """Calculates real-time 2nd innings win probabilities."""
    try:
        predictor = get_predictor()
        res = predictor.predict_win_probability(
            batting_team=req.batting_team,
            bowling_team=req.bowling_team,
            runs_scored=req.runs_scored,
            wickets_lost=req.wickets_lost,
            overs_completed=req.overs_completed,
            target_runs=req.target_runs
        )
        momentum = get_match_momentum_indicator(res["current_run_rate"], res["required_run_rate"])
        res["momentum_status"] = momentum["status"]
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/first-innings-score", response_model=FirstInningsScoreResponse, tags=["Predictions"])
def predict_first_innings_score(req: FirstInningsScoreRequest):
    """Predicts 1st innings total score projection and range."""
    try:
        predictor = get_predictor()
        res = predictor.predict_first_innings_score(
            batting_team=req.batting_team,
            bowling_team=req.bowling_team,
            runs_scored=req.runs_scored,
            wickets_lost=req.wickets_lost,
            overs_completed=req.overs_completed
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/simulate/scenario", tags=["Explainability & Simulation"])
def simulate_scenario(req: ScenarioSimulationRequest):
    """Simulates over-by-over win probability trajectories for what-if scenarios."""
    try:
        engine = get_explainability()
        trajectories = engine.simulate_what_if_trajectory(
            batting_team=req.batting_team,
            bowling_team=req.bowling_team,
            current_runs=req.current_runs,
            current_wickets=req.current_wickets,
            current_over=req.current_over,
            target_runs=req.target_runs,
            run_rates=req.run_rates,
            simulate_wickets_lost=req.simulate_wickets_lost
        )
        return {"trajectories": trajectories}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/model/feature-importance", tags=["Explainability & Simulation"])
def feature_importance():
    """Returns normalized tree feature importance values."""
    engine = get_explainability()
    return engine.get_feature_importances()


@app.post("/analytics/dls-par-score", response_model=DlsCalculationResponse, tags=["Cricket Analytics"])
def calculate_dls_par_score(req: DlsCalculationRequest):
    """Calculates Duckworth-Lewis-Stern (DLS) par score for rain interruptions."""
    par_score = estimate_dls_par_score(
        target_runs=req.target_runs,
        current_over=req.current_over,
        wickets_lost=req.wickets_lost
    )
    diff = req.current_runs - par_score
    return {
        "target_runs": req.target_runs,
        "current_over": req.current_over,
        "wickets_lost": req.wickets_lost,
        "current_runs": req.current_runs,
        "dls_par_score": par_score,
        "par_differential": diff,
        "is_ahead_of_par": diff >= 0
    }
