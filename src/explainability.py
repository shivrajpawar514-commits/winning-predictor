"""
Explainable AI (XAI) and Dynamic Scenario Simulation Engine for IPL Analytics.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from src.predictor import MatchPredictor
from src.config import CLASSIFICATION_FEATURES, REGRESSION_FEATURES


class ExplainabilityEngine:
    """Provides feature importance attribution and dynamic match simulation."""

    def __init__(self, predictor: MatchPredictor):
        self.predictor = predictor

    def get_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """Returns normalized feature importances for classifier and regressor."""
        clf_importances = {}
        if hasattr(self.predictor.clf, "feature_importances_"):
            raw_imp = self.predictor.clf.feature_importances_
            feature_names = [
                "Batting Team", "Bowling Team", "Cumulative Runs",
                "Wickets Lost", "Overs Completed", "Inning", "Target Runs"
            ]
            for name, imp in zip(feature_names, raw_imp):
                clf_importances[name] = round(float(imp), 4)

        reg_importances = {}
        if hasattr(self.predictor.reg, "feature_importances_"):
            raw_imp = self.predictor.reg.feature_importances_
            feature_names = [
                "Batting Team", "Bowling Team", "Cumulative Runs",
                "Wickets Lost", "Overs Completed"
            ]
            for name, imp in zip(feature_names, raw_imp):
                reg_importances[name] = round(float(imp), 4)

        return {
            "classification": clf_importances,
            "regression": reg_importances
        }

    def simulate_what_if_trajectory(
        self,
        batting_team: str,
        bowling_team: str,
        current_runs: int,
        current_wickets: int,
        current_over: float,
        target_runs: int,
        run_rates: List[float] = [6.0, 8.0, 10.0, 12.0],
        simulate_wickets_lost: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Simulates future over-by-over win probability trajectories
        under different hypothetical scoring rates and wicket scenarios.
        """
        trajectories = []
        overs_to_simulate = np.arange(int(current_over) + 1, 21)

        for rate in run_rates:
            points = []
            sim_runs = current_runs
            sim_wickets = min(10, current_wickets + simulate_wickets_lost)

            # Add starting point
            initial_res = self.predictor.predict_win_probability(
                batting_team=batting_team,
                bowling_team=bowling_team,
                runs_scored=current_runs,
                wickets_lost=current_wickets,
                overs_completed=current_over,
                target_runs=target_runs
            )
            points.append({
                "over": round(current_over, 1),
                "runs": current_runs,
                "wickets": current_wickets,
                "win_prob": round(initial_res["batting_win_probability"] * 100, 2)
            })

            for o in overs_to_simulate:
                if sim_runs >= target_runs:
                    points.append({
                        "over": float(o),
                        "runs": target_runs,
                        "wickets": sim_wickets,
                        "win_prob": 100.0
                    })
                    break
                if sim_wickets >= 10:
                    points.append({
                        "over": float(o),
                        "runs": sim_runs,
                        "wickets": 10,
                        "win_prob": 0.0
                    })
                    break

                sim_runs = int(min(target_runs, current_runs + (o - current_over) * rate))
                res = self.predictor.predict_win_probability(
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    runs_scored=sim_runs,
                    wickets_lost=sim_wickets,
                    overs_completed=float(o),
                    target_runs=target_runs
                )
                points.append({
                    "over": float(o),
                    "runs": sim_runs,
                    "wickets": sim_wickets,
                    "win_prob": round(res["batting_win_probability"] * 100, 2)
                })

            trajectories.append({
                "run_rate": rate,
                "label": f"Scoring at {rate} RPO (+{simulate_wickets_lost} wkts)",
                "points": points
            })

        return trajectories
