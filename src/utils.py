"""
Cricket Analytics & Match Utility Functions.
Includes Strike Rate, Economy Rate, and DLS Par Score estimation helpers.
"""

from typing import Dict, Any


def calculate_strike_rate(runs: int, balls_faced: int) -> float:
    """Calculates batting strike rate (runs per 100 balls)."""
    if balls_faced <= 0:
        return 0.0
    return round((runs / balls_faced) * 100.0, 2)


def calculate_economy_rate(runs_conceded: int, overs_bowled: float) -> float:
    """Calculates bowling economy rate (runs conceded per over)."""
    overs_int = int(overs_bowled)
    balls = int(round((overs_bowled - overs_int) * 10))
    balls = min(balls, 5)
    decimal_overs = overs_int + (balls / 6.0)
    
    if decimal_overs <= 0:
        return 0.0
    return round(runs_conceded / decimal_overs, 2)


def estimate_dls_par_score(target_runs: int, current_over: float, wickets_lost: int) -> int:
    """
    Estimates simple DLS-style resource-adjusted par score for rain interruptions
    based on standard 20-over resource decay curves.
    """
    total_resources = 100.0
    overs_int = int(current_over)
    balls = int(round((current_over - overs_int) * 10))
    decimal_overs = overs_int + (min(balls, 5) / 6.0)

    # Resource percentage curve
    overs_remaining_ratio = max(0.0, (20.0 - decimal_overs) / 20.0)
    wicket_penalty = (wickets_lost ** 1.3) * 3.5
    resources_remaining = max(0.0, (overs_remaining_ratio * 100.0) - wicket_penalty)
    resources_used = max(0.0, total_resources - resources_remaining)

    par_score = int(round((target_runs * (resources_used / 100.0))))
    return max(0, par_score)


def get_match_momentum_indicator(crr: float, rrr: float) -> Dict[str, Any]:
    """Computes qualitative momentum and match control rating."""
    diff = crr - rrr
    if diff >= 2.0:
        return {"status": "High Batting Dominance", "sentiment": "positive", "badge": "🔥 Strong Advantage"}
    elif diff >= 0.0:
        return {"status": "Batting Slight Advantage", "sentiment": "positive", "badge": "📈 Favorable"}
    elif diff >= -2.0:
        return {"status": "Tense Contest", "sentiment": "neutral", "badge": "⚖️ Balanced"}
    else:
        return {"status": "Bowling Dominance", "sentiment": "negative", "badge": "🛡️ Defending Advantage"}
