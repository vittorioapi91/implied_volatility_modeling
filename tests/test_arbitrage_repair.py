"""Tests for arbitrage-free repair."""

import numpy as np

from rates_models.arbitrage import validate_price_surface, validate_vol_surface
from rates_models.arbitrage_repair import repair_price_surface, repair_vol_surface_black76
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
