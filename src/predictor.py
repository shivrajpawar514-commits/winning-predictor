"""
Production inference engine for IPL Match Predictor.
Encapsulates model loading, input validation, and statistical predictions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
from src.config import (
    WIN_CLASSIFIER_PATH,
    RUNS_REGRESSOR_PATH,
    LE_BATTING_PATH,
    LE_BOWLING_PATH,
    METADATA_PATH
)
from src.feature_engineering import encode_single_match_state
from src.utils import validate_and_convert_overs, total_balls_to_decimal_overs


class MatchPredictor:
    """Production Inference Engine for IPL winning probabilities and run projections."""

    def __init__(
        self,
        classifier_path=None,
        regressor_path=None,
        le_batting_path=None,
        le_bowling_path=None,
        metadata_path=None
    ):
        classifier_path = Path(classifier_path or WIN_CLASSIFIER_PATH)
        regressor_path = Path(regressor_path or RUNS_REGRESSOR_PATH)
        le_batting_path = Path(le_batting_path or LE_BATTING_PATH)
        le_bowling_path = Path(le_bowling_path or LE_BOWLING_PATH)
        metadata_path = Path(metadata_path or METADATA_PATH)

        # Check for missing model files
        missing = [
            str(p) for p in [classifier_path, regressor_path, le_batting_path, le_bowling_path]
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required model artifact(s): {', '.join(missing)}. "
                f"Please train the models first using `python -m src.train` or `python train_model.py`."
            )

        try:
            self.clf = joblib.load(classifier_path)
            self.reg = joblib.load(regressor_path)
            self.le_batting = joblib.load(le_batting_path)
            self.le_bowling = joblib.load(le_bowling_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models from artifacts: {e}") from e

        self.teams: List[str] = sorted(list(self.le_batting.classes_))
        
        self.metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

    @classmethod
    def models_exist(
        cls,
        classifier_path=WIN_CLASSIFIER_PATH,
        regressor_path=RUNS_REGRESSOR_PATH,
        le_batting_path=LE_BATTING_PATH,
        le_bowling_path=LE_BOWLING_PATH
    ) -> bool:
        """Checks if all required model artifacts exist on disk."""
        return all(
            Path(p).exists()
            for p in [classifier_path, regressor_path, le_batting_path, le_bowling_path]
        )

    def validate_inputs(
        self,
        batting_team: str,
        bowling_team: str,
        runs_scored: int,
        wickets_lost: int,
        overs_completed: float,
        innings: int,
        target_runs: int = 0
    ) -> int:
        """
        Validates all match state parameters.
        Returns total legal balls bowled so far.
        """
        if batting_team not in self.teams:
            raise ValueError(f"Unknown batting team '{batting_team}'. Must be one of {self.teams}")
        if bowling_team not in self.teams:
            raise ValueError(f"Unknown bowling team '{bowling_team}'. Must be one of {self.teams}")
        if batting_team == bowling_team:
            raise ValueError("Batting team and Bowling team cannot be identical.")
        if not (0 <= runs_scored <= 500):
            raise ValueError(f"Runs scored {runs_scored} is out of realistic range [0, 500].")
        if not (0 <= wickets_lost <= 10):
            raise ValueError(f"Wickets lost {wickets_lost} must be in range [0, 10].")

        # Validate cricket overs correctly (e.g. 14.2 = 14 overs and 2 balls, not 14.2 decimal)
        total_balls = validate_and_convert_overs(overs_completed)

        if innings not in (1, 2):
            raise ValueError(f"Innings must be 1 or 2, got {innings}.")
        if innings == 2 and target_runs < 1:
            raise ValueError(f"Target runs must be positive for 2nd innings, got {target_runs}.")

        return total_balls

    def predict_win_probability(
        self,
        batting_team: str,
        bowling_team: str,
        runs_scored: int,
        wickets_lost: int,
        overs_completed: float,
        target_runs: int
    ) -> Dict[str, Any]:
        """Calculates 2nd innings win probability for both teams."""
        total_balls = self.validate_inputs(
            batting_team=batting_team,
            bowling_team=bowling_team,
            runs_scored=runs_scored,
            wickets_lost=wickets_lost,
            overs_completed=overs_completed,
            innings=2,
            target_runs=target_runs
        )

        runs_needed = max(0, target_runs - runs_scored)
        balls_remaining = max(0, 120 - total_balls)
        wickets_in_hand = max(0, 10 - wickets_lost)
        overs_remaining = balls_remaining / 6.0 if balls_remaining > 0 else 0.001
        rrr = runs_needed / overs_remaining if overs_remaining > 0 else 0.0

        state = encode_single_match_state(
            batting_team=batting_team,
            bowling_team=bowling_team,
            runs_scored=runs_scored,
            wickets_lost=wickets_lost,
            overs_completed=overs_completed,
            innings=2,
            target_runs=target_runs,
            le_batting=self.le_batting,
            le_bowling=self.le_bowling
        )

        # Boundary condition checks for definitive match outcomes
        if runs_scored >= target_runs:
            batting_win_prob = 1.0
            bowling_win_prob = 0.0
        elif wickets_lost >= 10 or (balls_remaining == 0 and runs_scored < target_runs):
            batting_win_prob = 0.0
            bowling_win_prob = 1.0
        else:
            probs = self.clf.predict_proba(state["features_cls"])[0]
            batting_win_prob = float(probs[1])
            bowling_win_prob = float(probs[0])

        return {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "batting_win_probability": batting_win_prob,
            "bowling_win_probability": bowling_win_prob,
            "current_run_rate": round(state["crr"], 2),
            "required_run_rate": round(rrr, 2),
            "runs_needed": runs_needed,
            "balls_remaining": balls_remaining,
            "wickets_in_hand": wickets_in_hand
        }

    def predict_first_innings_score(
        self,
        batting_team: str,
        bowling_team: str,
        runs_scored: int,
        wickets_lost: int,
        overs_completed: float
    ) -> Dict[str, Any]:
        """Predicts 1st innings final score projection and estimation range."""
        total_balls = self.validate_inputs(
            batting_team=batting_team,
            bowling_team=bowling_team,
            runs_scored=runs_scored,
            wickets_lost=wickets_lost,
            overs_completed=overs_completed,
            innings=1
        )

        state = encode_single_match_state(
            batting_team=batting_team,
            bowling_team=bowling_team,
            runs_scored=runs_scored,
            wickets_lost=wickets_lost,
            overs_completed=overs_completed,
            innings=1,
            target_runs=0,
            le_batting=self.le_batting,
            le_bowling=self.le_bowling
        )

        balls_remaining = max(0, 120 - total_balls)
        wickets_in_hand = max(0, 10 - wickets_lost)

        if wickets_lost >= 10 or balls_remaining == 0:
            projected_total = runs_scored
            lower_bound = runs_scored
            upper_bound = runs_scored
        else:
            predicted_runs = float(self.reg.predict(state["features_reg"])[0])
            projected_total = max(runs_scored, int(round(predicted_runs)))
            lower_bound = max(runs_scored, int(projected_total - 10))
            upper_bound = int(projected_total + 10)

        return {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "projected_total_runs": projected_total,
            "score_lower_bound": lower_bound,
            "score_upper_bound": upper_bound,
            "current_run_rate": round(state["crr"], 2),
            "balls_remaining": balls_remaining,
            "wickets_in_hand": wickets_in_hand
        }

