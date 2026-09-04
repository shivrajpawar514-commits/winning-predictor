"""
Unit tests for Cricket Analytics utility functions.
"""

import pytest
from src.utils import (
    calculate_strike_rate,
    calculate_economy_rate,
    estimate_dls_par_score,
    get_match_momentum_indicator
)


def test_strike_rate_calculation():
    assert calculate_strike_rate(50, 25) == 200.0
    assert calculate_strike_rate(0, 0) == 0.0
    assert calculate_strike_rate(45, 30) == 150.0


def test_economy_rate_calculation():
    assert calculate_economy_rate(30, 4.0) == 7.5
    assert calculate_economy_rate(0, 0.0) == 0.0


def test_estimate_dls_par_score():
    par_early = estimate_dls_par_score(target_runs=180, current_over=5.0, wickets_lost=0)
    par_late = estimate_dls_par_score(target_runs=180, current_over=15.0, wickets_lost=2)
    assert 0 < par_early < par_late <= 180


def test_get_match_momentum_indicator():
    res_pos = get_match_momentum_indicator(crr=11.0, rrr=8.0)
    assert res_pos["sentiment"] == "positive"
    
    res_neg = get_match_momentum_indicator(crr=6.0, rrr=11.0)
    assert res_neg["sentiment"] == "negative"
