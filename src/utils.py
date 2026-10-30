"""
Cricket Analytics & Match Utility Functions.
Includes Strike Rate, Economy Rate, Overs validation, and DLS Par Score estimation helpers.
"""

from typing import Dict, Any, Tuple


def validate_and_convert_overs(overs: float) -> int:
    """
    Validates cricket overs format and converts overs into total legal balls.
    Cricket notation: O.B where O is complete overs (0-20) and B is balls in current over (0-5).
    For example:
        14.2 -> 14 overs and 2 balls -> 14 * 6 + 2 = 86 total balls.
        20.0 -> 20 overs and 0 balls -> 120 total balls.

    Raises:
        TypeError: If overs is not a numeric value.
        ValueError: If overs is outside [0.0, 20.0], has ball count outside [0, 5],
                    or is formatted as an invalid decimal (e.g. 14.25, 14.8).
    """
    if not isinstance(overs, (int, float)):
        raise TypeError(f"Overs completed must be a number, got {type(overs).__name__}")

    if overs < 0.0 or overs > 20.0:
        raise ValueError(f"Overs completed {overs} is out of realistic range [0.0, 20.0].")

    overs_int = int(overs)
    # Check decimal precision to prevent treating arbitrary decimals (e.g. 14.25) as cricket overs
    frac = round(overs - overs_int, 4)
    ball_float = round(frac * 10, 3)
    ball_int = int(round(ball_float))

    if abs(ball_float - ball_int) > 1e-3:
        raise ValueError(
            f"Invalid cricket overs '{overs}'. Cricket overs must be formatted as O.B "
            f"(e.g., 14.2 for 14 overs and 2 balls). Decimal overs like {overs} are not valid cricket notation."
        )

    if ball_int > 5:
        raise ValueError(
            f"Invalid cricket overs '{overs}'. Ball count in an over must be between 0 and 5 (got {ball_int})."
        )

    if overs_int == 20 and ball_int > 0:
        raise ValueError(
            f"Invalid cricket overs '{overs}'. Maximum overs in a T20 innings is 20.0 (got {overs})."
        )

    total_balls = overs_int * 6 + ball_int
    return total_balls


def total_balls_to_overs(total_balls: int) -> float:
    """Converts total balls to cricket overs notation O.B (e.g. 86 balls -> 14.2)."""
    if total_balls < 0 or total_balls > 120:
        raise ValueError(f"Total balls {total_balls} must be between 0 and 120.")
    overs = total_balls // 6
    balls = total_balls % 6
    return round(overs + (balls / 10.0), 1)


def total_balls_to_decimal_overs(total_balls: int) -> float:
    """Converts total balls to mathematical decimal overs (e.g. 86 balls -> 14.3333...)."""
    if total_balls < 0:
        raise ValueError(f"Total balls {total_balls} cannot be negative.")
    return total_balls / 6.0


def calculate_strike_rate(runs: int, balls_faced: int) -> float:
    """Calculates batting strike rate (runs per 100 balls)."""
    if balls_faced <= 0:
        return 0.0
    return round((runs / balls_faced) * 100.0, 2)


def calculate_economy_rate(runs_conceded: int, overs_bowled: float) -> float:
    """Calculates bowling economy rate (runs conceded per 6 balls)."""
    total_balls = validate_and_convert_overs(overs_bowled)
    if total_balls <= 0:
        return 0.0
    decimal_overs = total_balls_to_decimal_overs(total_balls)
    return round(runs_conceded / decimal_overs, 2)


def estimate_dls_par_score(target_runs: int, current_over: float, wickets_lost: int) -> int:
    """
    Estimates simple DLS-style resource-adjusted par score for rain interruptions
    based on standard 20-over resource decay curves.
    """
    if target_runs < 1:
        raise ValueError(f"Target runs must be positive, got {target_runs}.")
    if not (0 <= wickets_lost <= 10):
        raise ValueError(f"Wickets lost {wickets_lost} must be in range [0, 10].")

    total_resources = 100.0
    total_balls = validate_and_convert_overs(current_over)
    decimal_overs = total_balls_to_decimal_overs(total_balls)

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

