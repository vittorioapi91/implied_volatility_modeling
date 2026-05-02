"""Tests for discrete no-arbitrage validation."""

import numpy as np
import pytest

from helper_module.arbitrage import (
    check_calls_butterfly,
    check_total_variance_along_tenor_columns,
    check_total_variance_calendar,
    validate_price_surface,
    validate_vol_surface,
    validate_vol_surface_per_expiry_black76,
)
from helper_module.black76 import black76_price


def test_valid_flat_vol_surface_passes() -> None:
    strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    Ts = np.array([0.5, 1.0, 2.0])
    forward = 0.03
    sigma = 0.25
    rows_s, rows_t, rows_sig = [], [], []
    for t in Ts:
        for k in strikes:
            rows_s.append(k)
            rows_t.append(t)
            rows_sig.append(sigma)
    rep = validate_vol_surface(
        np.array(rows_s),
        np.array(rows_t),
        np.array(rows_sig),
        forward=forward,
    )
    assert rep.ok
    assert rep.violations == []


def test_butterfly_violation_detected() -> None:
    strikes = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    p = np.array([black76_price(0.03, k, 1.0, 0.2, 1.0, "call") for k in strikes])
    p[2] += 0.01  # break convexity at interior
    bad = check_calls_butterfly(p, strikes, tol=1e-10)
    assert bad


def test_calendar_total_variance_violation() -> None:
    sig = np.array([[0.25, 0.25], [0.25, 0.05]])  # 2 strikes, 2 times
    T = np.array([1.0, 2.0])
    bad = check_total_variance_calendar(sig, T, tol=1e-12)
    assert bad


def test_sticky_delta_total_variance_column_violation() -> None:
    # Two tenors, one column: w = σ²T must increase in T
    sig = np.array([[0.5], [0.1]])  # w drops if T increases enough — pick T so w decreases
    T = np.array([1.0, 4.0])
    bad = check_total_variance_along_tenor_columns(sig, T, tol=1e-12)
    assert bad


def test_validate_vol_surface_per_expiry_matches_flat_rectangular() -> None:
    strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    Ts = np.array([0.5, 1.0])
    forward = 0.03
    sigma = 0.25
    mat = np.tile(strikes, (2, 1))
    sig = np.full_like(mat, sigma)
    rep_rect = validate_vol_surface(
        mat.ravel(),
        np.repeat(Ts, len(strikes)),
        sig.ravel(),
        forward=forward,
    )
    rep_row = validate_vol_surface_per_expiry_black76(mat, Ts, sig, forward=forward)
    assert rep_rect.ok == rep_row.ok
    assert len(rep_rect.violations) == len(rep_row.violations)
