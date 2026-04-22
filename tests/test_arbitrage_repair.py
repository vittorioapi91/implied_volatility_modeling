"""Tests for arbitrage-free repair."""

import numpy as np

from rates_models.arbitrage import validate_price_surface, validate_vol_surface
from rates_models.arbitrage import validate_vol_surface_per_expiry_black76
from rates_models.arbitrage_repair import (
    repair_price_surface,
    repair_vol_bergeron_grid_black76,
    repair_vol_surface_black76,
)
from rates_models.vae_vol_bergeron import TENORS_YEARS, strikes_for_bergeron_grid
from rates_models.black76 import black76_price


def test_repair_prices_demo_slice() -> None:
    strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    t = 1.0
    prices = np.array([black76_price(0.03, k, t, 0.25, 1.0, "call") for k in strikes])
    prices[2] += 0.002
    expiries = np.full_like(strikes, t, dtype=float)
    assert not validate_price_surface(strikes, expiries, prices).ok
    _, _, _, rep = repair_price_surface(strikes, expiries, prices)
    assert rep.ok


def test_repair_vol_calendar_break() -> None:
    strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    Ts = np.array([0.5, 1.0, 2.0])
    sigma_ok = 0.25
    sigma_bad_long = 0.08
    rows = []
    for t in Ts:
        s = sigma_bad_long if t >= 2.0 - 1e-9 else sigma_ok
        for k in strikes:
            rows.append((k, t, s))
    s = np.array([a for a, _, _ in rows])
    e = np.array([b for _, b, _ in rows])
    v = np.array([c for _, _, c in rows])
    assert not validate_vol_surface(s, e, v, 0.03).ok
    _, _, _, rep = repair_vol_surface_black76(s, e, v, 0.03, tol=1e-8)
    assert rep.ok


def test_repair_bergeron_grid_runs_and_often_feasible() -> None:
    """Bergeron-shaped repair alternates sticky variance and per-expiry price projection."""
    from rates_models.vae_vol_bergeron import make_synthetic_sabr_surfaces

    rng = np.random.default_rng(0)
    s = make_synthetic_sabr_surfaces(1, rng=rng)
    K = strikes_for_bergeron_grid(0.03)
    mat = s[0].reshape(len(TENORS_YEARS), K.shape[1])
    mat_r, rep = repair_vol_bergeron_grid_black76(
        mat, strikes=K, tenors=TENORS_YEARS, forward=0.03, max_iter=20
    )
    assert mat_r.shape == mat.shape
    assert np.all(np.isfinite(mat_r))
    v = validate_vol_surface_per_expiry_black76(K, TENORS_YEARS, mat_r, 0.03)
    assert rep.ok == v.ok
