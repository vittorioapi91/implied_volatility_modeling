"""Discrete LIBOR market model: simulation under spot LIBOR measure (Monte Carlo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rates_models.black76 import black76_price


@dataclass(frozen=True)
class LMMConfig:
    """Piecewise-constant LIBORs on tenor structure T_0,...,T_n."""

    forwards: np.ndarray  # shape (n,) — L_i(0) for i = 0..n-1 between T_i and T_{i+1}
    tau: np.ndarray  # shape (n,) — year fractions tau_i = T_{i+1} - T_i
    vols: np.ndarray  # shape (n,) — constant sigma_i for each forward
    corr: np.ndarray  # shape (n, n) — correlation of driving Brownian motions


def simulate_lmm_spot_measure(
    cfg: LMMConfig,
    horizon: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate instantaneous forward LIBORs L_i(t) under the **spot LIBOR measure**.

    Uses log-Euler on positive LIBORs; drift follows the standard spot-LMM drift
    (see Brigo–Mercurio / Glasserman).

    Returns array of shape (n_paths, n_steps + 1, n) with L at each time grid point.
    """
    rng = rng or np.random.default_rng()
    forwards = np.asarray(cfg.forwards, dtype=float)
    tau = np.asarray(cfg.tau, dtype=float)
    sigma = np.asarray(cfg.vols, dtype=float)
    rho = np.asarray(cfg.corr, dtype=float)
    n = forwards.shape[0]
    if tau.shape != (n,) or sigma.shape != (n,) or rho.shape != (n, n):
        raise ValueError("forwards, tau, vols, and corr must have compatible shapes.")

    chol = np.linalg.cholesky(rho)
    dt = horizon / n_steps
    sqrt_dt = np.sqrt(dt)

    # tenor dates: assume T_0 = 0, T_{i+1} = T_i + tau_i
    t_dates = np.concatenate([[0.0], np.cumsum(tau)])

    out = np.zeros((n_paths, n_steps + 1, n))
    out[:, 0, :] = forwards

    inc = rng.standard_normal((n_paths, n_steps, n))

    for step in range(n_steps):
        t = step * dt
        # eta(t): Brigo–Mercurio index with T_{eta-1} <= t < T_{eta}; drift sums j = eta..i
        eta = int(np.searchsorted(t_dates, t, side="right")) - 1
        eta = min(max(eta, 0), n - 1)

        L = out[:, step, :].copy()
        Z = inc[:, step, :] @ chol.T

        drift = np.zeros((n_paths, n))
        for i in range(eta, n):
            acc = np.zeros(n_paths)
            for j in range(eta, i + 1):
                denom = 1.0 + tau[j] * L[:, j]
                acc += tau[j] * sigma[i] * sigma[j] * rho[i, j] * L[:, j] / denom
            drift[:, i] = -acc

        # Log-Euler: L(t+dt) = L * exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z)
        mu = drift
        growth = (mu - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z
        np.clip(L, 1e-12, None, out=L)
        out[:, step + 1, :] = L * np.exp(growth)

    return out


def price_caplet_lmm_mc(
    cfg: LMMConfig,
    caplet_index: int,
    rng: np.random.Generator | None = None,
    *,
    n_paths: int = 50_000,
) -> tuple[float, float]:
    """
    Monte Carlo ATM caplet on L_k under the **T_{k+1}-forward measure** Q^{k+1}.

    There, L_k is a martingale (lognormal with vol sigma_k), so the caplet matches
    Black-76 with expiry T_k = sum_{j<k} tau_j. This is the standard LMM caplet
    price; raw averaging under the spot measure would require deflation by the
    rolling numeraire.

    Returns (mc_price, black76_price).
    """
    rng = rng or np.random.default_rng()
    k = caplet_index
    tau = float(cfg.tau[k])
    t_reset = float(np.sum(cfg.tau[:k]))
    if t_reset <= 0.0:
        raise ValueError("caplet_index must be >= 1 so fixing time T_k > 0.")

    f0 = float(cfg.forwards[k])
    sig = float(cfg.vols[k])
    z = rng.standard_normal(n_paths)
    lk = f0 * np.exp(-0.5 * sig * sig * t_reset + sig * np.sqrt(t_reset) * z)
    payoff = tau * np.maximum(lk - f0, 0.0)
    mc = float(np.mean(payoff))

    black = tau * black76_price(
        forward=f0,
        strike=f0,
        time_to_expiry=t_reset,
        sigma=sig,
        discount=1.0,
        option_type="call",
    )
    return mc, black
