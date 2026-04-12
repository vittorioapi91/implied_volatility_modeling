"""One-factor Gaussian HJM: forward-rate simulation and caplet vol (Black equivalence)."""

from __future__ import annotations

import math

import numpy as np


def deterministic_vol_hjm(
    t: np.ndarray,
    T: np.ndarray,
    sigma0: float,
    kappa: float,
) -> np.ndarray:
    """
    Separable deterministic volatility often used in Gaussian HJM / Hull–White:

        sigma(t, T) = sigma0 * exp(-kappa * (T - t))  for T >= t, else 0.

    Shapes broadcast: t shape (nt,), T shape (nT,) -> output (nt, nT).
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    T = np.asarray(T, dtype=float).reshape(1, -1)
    mask = T >= t
    vol = sigma0 * np.exp(-kappa * (T - t))
    return np.where(mask, vol, 0.0)


def simulate_instantaneous_forwards_gaussian_hjm(
    f0: np.ndarray,
    times: np.ndarray,
    maturity_grid: np.ndarray,
    sigma0: float,
    kappa: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate instantaneous forward rates f(t,T) under risk-neutral Gaussian HJM
    with deterministic volatility sigma(t,T) = sigma0 * exp(-kappa*(T-t)).

    Uses discretized SDE:
        df(t,T) = sigma(t,T) * (int_t^T sigma(t,u) du) dt + sigma(t,T) dW_t

    f0: forward curve f(0,T) on `maturity_grid`
    times: simulation times
    Returns array shape (n_times, n_maturities) with f(t, T_j).
    """
    rng = rng or np.random.default_rng()
    f0 = np.asarray(f0, dtype=float).ravel()
    Tgrid = np.asarray(maturity_grid, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if f0.shape != Tgrid.shape:
        raise ValueError("f0 and maturity_grid must match.")

    n_t = times.size
    n_T = Tgrid.size
    f = np.zeros((n_t, n_T))
    f[0, :] = f0

    for i in range(1, n_t):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            raise ValueError("times must be strictly increasing.")
        t_prev = times[i - 1]
        z = rng.standard_normal()

        sig = sigma0 * np.exp(-kappa * (Tgrid - t_prev))
        sig = np.where(Tgrid >= t_prev, sig, 0.0)
        # int_{t}^{T} sigma(t,u) du = sigma0/kappa * (e^{-kappa(T-t)} - 1) for T>=t
        integral = np.where(
            Tgrid >= t_prev,
            (sigma0 / kappa) * (np.exp(-kappa * (Tgrid - t_prev)) - 1.0),
            0.0,
        )
        drift = sig * integral
        f[i, :] = f[i - 1, :] + drift * dt + sig * math.sqrt(dt) * z

    return f


def caplet_black_vol_from_gaussian_hjm(
    T: float,
    S: float,
    sigma0: float,
    kappa: float,
) -> float:
    """
    Black implied (lognormal) volatility for a caplet on [T, S] under one-factor
    Gaussian HJM with sigma(t,T)=sigma0*exp(-kappa*(T-t)).

    Closed form from integrated variance of the forward rate over [0, T].
    """
    if T <= 0.0 or S <= T:
        raise ValueError("Need 0 < T < S (caplet reset at T on period ending S).")

    tau = S - T
    # L(t) ~ lognormal under T-forward measure in HW/Gaussian HJM;
    # Black vol^2 * T = Var[log L(T)] in the standard reduction — use HW caplet formula:
    # sigma_B^2 = (sigma0^2 / (2 kappa^3 tau^2)) * (1 - e^{-kappa T})^2 * (1 - e^{-2 kappa tau})
    num = (sigma0**2) * (1.0 - math.exp(-kappa * T)) ** 2 * (1.0 - math.exp(-2.0 * kappa * tau))
    den = 2.0 * (kappa**3) * (tau**2)
    return math.sqrt(num / den / T)
