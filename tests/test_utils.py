"""
Unit tests for Cricket Analytics utility functions and overs validation.
"""

import pytest
from src.utils import (
    calculate_strike_rate,
    calculate_economy_rate,
    estimate_dls_par_score,
    get_match_momentum_indicator,
    validate_and_convert_overs,
    total_balls_to_overs,
    total_balls_to_decimal_overs
)


def test_validate_and_convert_overs_valid():
    # 14.2 means 14 overs and 2 balls -> 14*6 + 2 = 86 balls
    assert validate_and_convert_overs(14.2) == 86
    assert validate_and_convert_overs(0.0) == 0
    assert validate_and_convert_overs(0.1) == 1
    assert validate_and_convert_overs(0.5) == 5
    assert validate_and_convert_overs(1.0) == 6
    assert validate_and_convert_overs(19.5) == 119
    assert validate_and_convert_overs(20.0) == 120


def test_validate_and_convert_overs_invalid_ball_count():
    # 14.6 means 6 balls in current over, which is invalid cricket notation (over complete = 15.0)
    with pytest.raises(ValueError, match="Ball count in an over must be between 0 and 5"):
        validate_and_convert_overs(14.6)

    with pytest.raises(ValueError, match="Ball count in an over must be between 0 and 5"):
        validate_and_convert_overs(14.8)


def test_validate_and_convert_overs_invalid_decimals():
    # 14.25 is not cricket notation O.B
    with pytest.raises(ValueError, match="Cricket overs must be formatted as O.B"):
        validate_and_convert_overs(14.25)

    with pytest.raises(ValueError, match="Cricket overs must be formatted as O.B"):
        validate_and_convert_overs(14.333)


def test_validate_and_convert_overs_boundary_violations():
    with pytest.raises(ValueError, match="out of realistic range"):
        validate_and_convert_overs(-0.1)

    with pytest.raises(ValueError, match="out of realistic range"):
        validate_and_convert_overs(20.5)

    with pytest.raises(ValueError, match="out of realistic range"):
        validate_and_convert_overs(20.1)



def test_validate_and_convert_overs_type_error():
    with pytest.raises(TypeError, match="must be a number"):
        validate_and_convert_overs("14.2")


def test_total_balls_to_overs():
    assert total_balls_to_overs(86) == 14.2
    assert total_balls_to_overs(0) == 0.0
    assert total_balls_to_overs(120) == 20.0
    assert total_balls_to_overs(5) == 0.5


def test_total_balls_to_decimal_overs():
    assert pytest.approx(total_balls_to_decimal_overs(86), 0.001) == 86 / 6.0
    assert total_balls_to_decimal_overs(120) == 20.0
    assert total_balls_to_decimal_overs(0) == 0.0


def test_strike_rate_calculation():
    assert calculate_strike_rate(50, 25) == 200.0
    assert calculate_strike_rate(0, 0) == 0.0
    assert calculate_strike_rate(45, 30) == 150.0


def test_economy_rate_calculation():
    assert calculate_economy_rate(30, 4.0) == 7.5
    assert calculate_economy_rate(0, 0.0) == 0.0
    # 14.2 overs = 86 balls -> decimal overs = 86 / 6 = 14.333 -> 120 / 14.333 = 8.37
    assert calculate_economy_rate(120, 14.2) == 8.37


def test_estimate_dls_par_score():
    par_early = estimate_dls_par_score(target_runs=180, current_over=5.0, wickets_lost=0)
    par_late = estimate_dls_par_score(target_runs=180, current_over=15.0, wickets_lost=2)
    assert 0 < par_early < par_late <= 180


def test_estimate_dls_par_score_invalid():
    with pytest.raises(ValueError, match="Target runs must be positive"):
        estimate_dls_par_score(target_runs=0, current_over=10.0, wickets_lost=2)
    with pytest.raises(ValueError, match="Wickets lost"):
        estimate_dls_par_score(target_runs=180, current_over=10.0, wickets_lost=11)


def test_get_match_momentum_indicator():
    res_pos = get_match_momentum_indicator(crr=11.0, rrr=8.0)
    assert res_pos["sentiment"] == "positive"
    
    res_neg = get_match_momentum_indicator(crr=6.0, rrr=11.0)
    assert res_neg["sentiment"] == "negative"

