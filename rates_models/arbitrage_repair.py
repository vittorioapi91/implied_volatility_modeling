"""
Repair option prices / Black–76 vol grids toward the closest arbitrage-free surface
on a finite grid (in the senses implemented in :mod:`rates_models.arbitrage`).

**Per-expiry call prices** (fixed T): Euclidean (L²) projection onto the polyhedron
defined by discrete strike monotonicity and butterfly (convexity) inequalities.
This is a convex quadratic program solved with ``scipy.optimize.minimize`` (SLSQP).

**Calendar (total variance)**: along each strike row, replace w(K,·) = σ²T by its
cumulative maximum in T (isotonic regression with increasing constraint). This is
the smallest pointwise increase to w that restores monotonicity in T (in ℓ∞ on
the cumulative-sum slack, not L²—see notebook).

**Vol surface**: alternate calendar repair on w and per-expiry price projection with
Black–76 inversion until checks pass or ``max_iter`` is reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, minimize

from rates_models.arbitrage import (
    _grid_from_long,
    validate_price_surface,
    validate_vol_surface,
    validate_vol_surface_per_expiry_black76,
)
from rates_models.black76 import black76_implied_vol, black76_price


@dataclass
class RepairResult:
    """Outcome of a repair step."""

    ok: bool
    l2_delta: float
    max_abs_change: float
    iterations: int = 1
    report_after: object | None = None
    details: dict[str, object] = field(default_factory=dict)


def project_call_prices_l2(
    p0: np.ndarray,
    *,
    tol: float = 1e-14,
    bounds_lo: float = 0.0,
    bounds_hi: float | None = None,
) -> np.ndarray:
    """
    Closest vector (Euclidean) to ``p0`` satisfying, for interior indices:
    - C[i] >= C[i+1] + tol (calls decreasing in strike)
    - C[i-1] - 2*C[i] + C[i+1] >= tol (discrete convexity / butterfly)

    If ``bounds_hi`` is None, use ``max(p0) * 2 + 0.01`` as an upper bound.
    """
    p0 = np.asarray(p0, dtype=float).ravel()
    n = p0.size
    if n <= 1:
        out = p0.copy()
        if bounds_hi is not None:
            np.clip(out, bounds_lo, bounds_hi, out=out)
        return out

    hi = bounds_hi if bounds_hi is not None else float(np.max(p0) * 2.0 + 0.01)
    bnds = Bounds(bounds_lo, hi)

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - p0) ** 2))

    cons: list[dict] = []
    for i in range(n - 1):
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, i=i: x[i] - x[i + 1] - tol,
            }
        )
    for i in range(1, n - 1):
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, i=i: x[i - 1] - 2.0 * x[i] + x[i + 1] - tol,
            }
        )

    x0 = np.clip(p0, bounds_lo, hi)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"L2 projection failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def cumulative_max_total_variance(w: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Replace each row (default axis=1 = time) by cumulative maximum so that
    w[..., j] >= w[..., j-1] elementwise along that axis.
    """
    return np.maximum.accumulate(w, axis=axis)


def repair_price_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    prices: np.ndarray,
    *,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RepairResult]:
    """
    Long-format arrays. Returns repaired (strikes, expiries, prices_long) and metrics.
    """
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    prices = np.asarray(prices, dtype=float)
    u_k, u_t, pmat = _grid_from_long(strikes, expiries, prices)
    p_new = np.empty_like(pmat)
    hi = float(np.nanmax(pmat) * 2.0 + 0.05)
    for j in range(u_t.size):
        sl = np.argsort(u_k)
        pcol = pmat[sl, j]
        p_new[sl, j] = project_call_prices_l2(pcol, tol=max(tol * 1e-4, 1e-14), bounds_hi=hi)

    delta = p_new - pmat
    strikes_out, expiries_out, prices_out = _long_from_matrix(u_k, u_t, p_new)
    rep = RepairResult(
        ok=validate_price_surface(strikes_out, expiries_out, prices_out, tol=tol).ok,
        l2_delta=float(np.sqrt(np.sum(delta**2))),
        max_abs_change=float(np.max(np.abs(delta))),
        iterations=1,
        report_after=validate_price_surface(strikes_out, expiries_out, prices_out, tol=tol),
    )
    return strikes_out, expiries_out, prices_out, rep


def _long_from_matrix(
    u_k: np.ndarray, u_t: np.ndarray, mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    for j, t in enumerate(u_t):
        for i, k in enumerate(u_k):
            rows.append((k, t, mat[i, j]))
    s = np.array([a for a, _, _ in rows])
    e = np.array([b for _, b, _ in rows])
    p = np.array([c for _, _, c in rows])
    return s, e, p


def repair_vol_surface_black76(
    strikes: np.ndarray,
    expiries: np.ndarray,
    sigma: np.ndarray,
    forward: float,
    discount: float = 1.0,
    *,
    tol: float = 1e-9,
    max_iter: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RepairResult]:
    """
    Repair implied Black–76 vol grid: alternate (i) cumulative max on w=σ²T per strike row,
    (ii) L² projection of Black prices per expiry column, (iii) reinvert σ from prices.

    Returns repaired (strikes, expiries, sigma_long) and :class:`RepairResult`.
    """
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    sigma0 = np.asarray(sigma, dtype=float)
    u_k, u_t, sig = _grid_from_long(strikes, expiries, sigma0)
    sig_init = sig.copy()
    sig_work = sig.copy()
    # Loose upper bound on Black call values under repair
    hi_price = max(1.0, float(forward) * 2.0 * discount + 0.1)

    it = 0
    while it < max_iter:
        it += 1
        T_row = u_t.reshape(1, -1)
        w = (sig_work**2) * T_row
        w = cumulative_max_total_variance(w, axis=1)
        sig_work = np.sqrt(np.maximum(w / T_row, 0.0))

        pmat = np.zeros_like(sig_work)
        for j, t in enumerate(u_t):
            for i, k in enumerate(u_k):
                pmat[i, j] = black76_price(forward, k, t, sig_work[i, j], discount, "call")

        for j in range(u_t.size):
            sl = np.argsort(u_k)
            pcol = pmat[sl, j]
            pmat[sl, j] = project_call_prices_l2(
                pcol, tol=max(tol * 1e-4, 1e-14), bounds_hi=hi_price
            )

        for j, t in enumerate(u_t):
            for i, k in enumerate(u_k):
                sig_work[i, j] = black76_implied_vol(
                    forward, k, t, pmat[i, j], discount, "call"
                )

        sig_long = sig_work.ravel(order="F")
        s_chk, e_chk, _ = _long_from_matrix(u_k, u_t, sig_work)
        rep = validate_vol_surface(s_chk, e_chk, sig_long, forward, discount, tol=tol)
        if rep.ok:
            s, e, sig_out = _long_from_matrix(u_k, u_t, sig_work)
            l2_sigma = float(np.sqrt(np.sum((sig_work - sig_init) ** 2)))
            return (
                s,
                e,
                sig_out,
                RepairResult(
                    ok=True,
                    l2_delta=l2_sigma,
                    max_abs_change=float(np.max(np.abs(sig_work - sig_init))),
                    iterations=it,
                    report_after=rep,
                    details={"sigma_matrix": sig_work, "strikes": u_k, "expiries": u_t},
                ),
            )

    s, e, sig_out = _long_from_matrix(u_k, u_t, sig_work)
    final = validate_vol_surface(s, e, sig_out, forward, discount, tol=tol)
    l2_sigma = float(np.sqrt(np.sum((sig_work - sig_init) ** 2)))
    return (
        s,
        e,
        sig_out,
        RepairResult(
            ok=final.ok,
            l2_delta=l2_sigma,
            max_abs_change=float(np.max(np.abs(sig_work - sig_init))),
            iterations=it,
            report_after=final,
            details={"sigma_matrix": sig_work, "strikes": u_k, "expiries": u_t},
        ),
    )


def repair_vol_bergeron_grid_black76(
    sigma: np.ndarray,
    *,
    strikes: np.ndarray,
    tenors: np.ndarray,
    forward: float,
    discount: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 25,
) -> tuple[np.ndarray, RepairResult]:
    """
    Repair an 8×5 style vol grid where **strikes vary by row** (tenor), as in the Bergeron
    VAE demo. Alternates: (i) cumulative maximum on sticky-delta total variance
    :math:`w=\\sigma^2 T` along tenor within each delta column, (ii) per-expiry L² projection
    of Black–76 call prices onto the discrete no-arbitrage polyhedron, with implied-vol
    reinversion.

    ``sigma``, ``strikes`` share shape ``(n_tenors, n_deltas)``; ``tenors`` is ``(n_tenors,)``.
    """
    sigma_work = np.asarray(sigma, dtype=float).copy()
    strikes = np.asarray(strikes, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    if sigma_work.shape != strikes.shape:
        raise ValueError("sigma and strikes must have the same shape.")
    if tenors.ndim != 1 or tenors.shape[0] != sigma_work.shape[0]:
        raise ValueError("tenors must be 1-D with length sigma.shape[0].")

    n_t, n_d = sigma_work.shape
    sig_init = sigma_work.copy()
    hi_price = max(1.0, float(forward) * 2.0 * discount + 0.1)
    Tcol = tenors.reshape(-1, 1)

    it = 0
    last_rep = validate_vol_surface_per_expiry_black76(
        strikes, tenors, sigma_work, forward, discount, tol=tol
    )
    while it < max_iter:
        it += 1
        w = (sigma_work**2) * Tcol
        w = np.maximum.accumulate(w, axis=0)
        sigma_work = np.sqrt(np.maximum(w / Tcol, 1e-18))

        for i in range(n_t):
            order = np.argsort(strikes[i])
            Ks = strikes[i, order]
            sg = sigma_work[i, order]
            t = float(tenors[i])
            prices = np.array(
                [
                    black76_price(forward, float(Ks[j]), t, float(sg[j]), discount, "call")
                    for j in range(n_d)
                ]
            )
            try:
                p_proj = project_call_prices_l2(
                    prices,
                    tol=max(tol * 1e-4, 1e-14),
                    bounds_hi=hi_price,
                )
            except RuntimeError:
                p_proj = prices
            sig_row = np.empty(n_d, dtype=float)
            for j in range(n_d):
                try:
                    sig_row[j] = black76_implied_vol(
                        forward,
                        float(Ks[j]),
                        t,
                        float(p_proj[j]),
                        discount,
                        "call",
                        bracket=(1e-10, 10.0),
                    )
                except ValueError:
                    sig_row[j] = float(sg[j])
            sigma_work[i, order] = sig_row

        last_rep = validate_vol_surface_per_expiry_black76(
            strikes, tenors, sigma_work, forward, discount, tol=tol
        )
        if last_rep.ok:
            l2_sigma = float(np.sqrt(np.sum((sigma_work - sig_init) ** 2)))
            return (
                sigma_work,
                RepairResult(
                    ok=True,
                    l2_delta=l2_sigma,
                    max_abs_change=float(np.max(np.abs(sigma_work - sig_init))),
                    iterations=it,
                    report_after=last_rep,
                    details={"sigma_matrix": sigma_work, "strikes": strikes, "tenors": tenors},
                ),
            )

    l2_sigma = float(np.sqrt(np.sum((sigma_work - sig_init) ** 2)))
    return (
        sigma_work,
        RepairResult(
            ok=last_rep.ok,
            l2_delta=l2_sigma,
            max_abs_change=float(np.max(np.abs(sigma_work - sig_init))),
            iterations=it,
            report_after=last_rep,
            details={"sigma_matrix": sigma_work, "strikes": strikes, "tenors": tenors},
        ),
    )
