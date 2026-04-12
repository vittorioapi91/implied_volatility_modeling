"""
Discrete no-arbitrage checks for Black–76 option prices and implied-vol grids.

Checks implemented (necessary conditions on a finite grid):

**Prices (per expiry, calls on a forward)**  
- Monotonicity in strike: C(K1) ≥ C(K2) for K1 < K2.  
- Convexity (butterfly): C(K−) − 2C(K) + C(K+) ≥ 0 on interior strikes.

**Volatility surface → prices**  
- Same as above on Black–76 prices implied by σ(K,T).  
- Calendar / total variance: w(K,T) = σ(K,T)² T non-decreasing in T at each strike
  (standard necessary condition for absence of calendar arbitrage in total-variance terms).

These are standard discrete diagnostics; they do not replace full continuum Dupire-type
constraints on an interpolated surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rates_models.black76 import black76_price


@dataclass
class ArbitrageReport:
    ok: bool
    violations: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


def _grid_from_long(
    strikes: np.ndarray,
    expiries: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted unique strikes/expries and 2D value matrix (nK, nT)."""
    u_strikes = np.unique(strikes)
    u_expiries = np.unique(expiries)
    mat = np.full((u_strikes.size, u_expiries.size), np.nan)
    si = {float(s): i for i, s in enumerate(u_strikes)}
    ti = {float(t): j for j, t in enumerate(u_expiries)}
    for s, t, v in zip(strikes, expiries, values, strict=True):
        mat[si[float(s)], ti[float(t)]] = float(v)
    if np.any(np.isnan(mat)):
        raise ValueError("Duplicate or missing (strike, expiry) pairs in grid.")
    return u_strikes, u_expiries, mat


def check_calls_strike_monotonicity(
    prices: np.ndarray,
    strikes: np.ndarray,
    tol: float = 1e-10,
) -> list[str]:
    """
    For fixed expiry: call price must decrease in strike.
    prices, strikes same length, strikes sorted ascending.
    """
    if prices.size < 2:
        return []
    bad = prices[:-1] + tol < prices[1:]
    idx = np.where(bad)[0]
    out: list[str] = []
    for i in idx:
        out.append(
            f"Strike monotonicity: C(K={strikes[i]:.6g}) < C(K={strikes[i + 1]:.6g}) "
            f"({prices[i]:.6g} < {prices[i + 1]:.6g})."
        )
    return out


def check_calls_butterfly(
    prices: np.ndarray,
    strikes: np.ndarray,
    tol: float = 1e-10,
) -> list[str]:
    """
    Discrete convexity: C[i-1] - 2C[i] + C[i+1] >= 0 for interior i.
    """
    if prices.size < 3:
        return []
    d2 = prices[:-2] - 2.0 * prices[1:-1] + prices[2:]
    bad = d2 < -tol
    idx = np.where(bad)[0]
    out: list[str] = []
    for j, i in enumerate(idx):
        k0, k1, k2 = strikes[i], strikes[i + 1], strikes[i + 2]
        out.append(
            f"Butterfly at K={k1:.6g}: C({k0:.6g})-2C({k1:.6g})+C({k2:.6g})={d2[i]:.6g} < 0."
        )
    return out


def check_total_variance_calendar(
    sigma: np.ndarray,
    expiries: np.ndarray,
    tol: float = 1e-12,
) -> list[str]:
    """
    sigma: shape (nK, nT), expiries sorted ascending.
    w(K,T) = σ² T should be non-decreasing in T for each strike row.
    """
    if sigma.shape[1] < 2:
        return []
    T = expiries.reshape(1, -1)
    w = (sigma**2) * T
    diff = np.diff(w, axis=1)
    out: list[str] = []
    for c in range(diff.shape[1]):
        bad_rows = np.where(diff[:, c] < -tol)[0]
        if bad_rows.size == 0:
            continue
        r0 = int(bad_rows[0])
        out.append(
            "Calendar (total var): w=σ²T decreases between "
            f"T={expiries[c]:.6g} and T={expiries[c + 1]:.6g} "
            f"for {bad_rows.size} strike(s) "
            f"(example row {r0}: w={w[r0, c]:.6g} → {w[r0, c + 1]:.6g})."
        )
    return out


def validate_price_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    prices: np.ndarray,
    tol: float = 1e-10,
) -> ArbitrageReport:
    """
    Long-format arrays of equal length. Validates per expiry slice, then calendar on prices
    (C(K,T2) >= C(K,T1) for T2>T1 is NOT enforced here for rates; use validate_vol_surface
    for total-variance calendar on σ).
    """
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    prices = np.asarray(prices, dtype=float)
    u_k, u_t, pmat = _grid_from_long(strikes, expiries, prices)

    violations: list[str] = []
    for j, t in enumerate(u_t):
        sl = np.argsort(u_k)
        ks = u_k[sl]
        pv = pmat[sl, j]
        violations.extend(
            [f"[T={t:.6g}] {m}" for m in check_calls_strike_monotonicity(pv, ks, tol)]
        )
        violations.extend(
            [f"[T={t:.6g}] {b}" for b in check_calls_butterfly(pv, ks, tol)]
        )

    # We do not enforce C(K,T2) ≥ C(K,T1) for T2>T1 on raw prices: that is not
    # required for all rate option surfaces. Calendar constraints on σ are
    # handled via total variance in validate_vol_surface.

    return ArbitrageReport(ok=len(violations) == 0, violations=violations, details={"strikes": u_k, "expiries": u_t})


def validate_vol_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    sigma: np.ndarray,
    forward: float,
    discount: float = 1.0,
    tol: float = 1e-10,
) -> ArbitrageReport:
    """
    Black–76 implied vol grid (same units as black76_price). Builds call prices and runs
    strike checks plus total-variance calendar in T.
    """
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    u_k, u_t, sig_mat = _grid_from_long(strikes, expiries, sigma)

    prices_flat: list[float] = []
    strikes_flat: list[float] = []
    expiries_flat: list[float] = []
    for j, t in enumerate(u_t):
        for i, k in enumerate(u_k):
            p = black76_price(forward, k, t, sig_mat[i, j], discount, "call")
            prices_flat.append(p)
            strikes_flat.append(k)
            expiries_flat.append(t)

    price_rep = validate_price_surface(
        np.array(strikes_flat),
        np.array(expiries_flat),
        np.array(prices_flat),
        tol=tol,
    )
    violations = list(price_rep.violations)

    cal = check_total_variance_calendar(sig_mat, u_t, tol=max(tol, 1e-14))
    violations.extend(cal)

    return ArbitrageReport(
        ok=len(violations) == 0,
        violations=violations,
        details={
            "strikes": u_k,
            "expiries": u_t,
            "sigma": sig_mat,
            "prices": np.array(prices_flat).reshape(u_k.size, u_t.size),
        },
    )
