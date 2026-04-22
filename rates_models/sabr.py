"""SABR implied volatility (Hagan et al. 2002 asymptotic expansion)."""

from __future__ import annotations

import math


def sabr_implied_vol_lognormal(
    forward: float,
    strike: float,
    time_to_expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    Black (lognormal) implied volatility under SABR.

    Uses the common Hagan et al. formula (beta in [0,1]). For beta=0 or beta=1,
    specialized limits are used to avoid 0/0.
    """
    if time_to_expiry <= 0.0:
        return alpha * max(forward, 1e-12) ** (beta - 1.0) if beta < 1.0 else alpha

    eps = 1e-14
    f = max(forward, eps)
    k = max(strike, eps)
    log_fk = math.log(f / k)

    if abs(beta) < 1e-12:
        # beta = 0: normal SABR — caller may prefer sabr_normal_vol; this returns *Black* vol
        # Hagan normal expansion gives sigma_N; convert approximately via ATM: sigma_B ≈ sigma_N / F
        sigma_n = sabr_normal_vol(forward, strike, time_to_expiry, alpha, rho, nu)
        atm = max(0.5 * (f + k), eps)
        return sigma_n / atm

    one_minus_beta = 1.0 - beta
    fk_pow = (f * k) ** (0.5 * one_minus_beta)
    # Standard Hagan denominator in log-moneyness.
    den = fk_pow * (
        1.0
        + (one_minus_beta**2 / 24.0) * log_fk**2
        + (one_minus_beta**4 / 1920.0) * log_fk**4
    )
    term1 = alpha / den

    z = (nu / alpha) * fk_pow * log_fk

    inner = (
        1.0
        + (
            (one_minus_beta**2 / 24.0) * alpha**2 / ((f * k) ** one_minus_beta)
            + 0.25 * rho * beta * nu * alpha / fk_pow
            + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
        )
        * time_to_expiry
    )
    # ATM branch uses analytic z/x -> 1 limit.
    if abs(log_fk) < 1e-10:
        sigma_atm = alpha / (f ** one_minus_beta)
        return sigma_atm * inner
    return term1 * _z_over_x(z, rho) * inner


def sabr_normal_vol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    alpha: float,
    rho: float,
    nu: float,
) -> float:
    """SABR with beta=0: implied *normal* volatility (Bachelier vol)."""
    if time_to_expiry <= 0.0:
        return alpha

    eps = 1e-14
    f = forward
    k = strike
    fk_mid = 0.5 * (f + k)
    log_fk = math.log(f / k) if f > eps and k > eps else 0.0

    term1 = alpha * (f - k) / log_fk if abs(log_fk) > 1e-12 else alpha
    term1 *= (1.0 + (1.0 / 24.0) * log_fk**2 + (1.0 / 1920.0) * log_fk**4) / (
        1.0 + (1.0 / 24.0) * (math.log(f * k + eps)) ** 2 + (1.0 / 1920.0) * (math.log(f * k + eps)) ** 4
    )

    z = nu / alpha * (f - k)
    inner = 1.0 + (-(1.0 / 3.0) * rho**2 + 0.25) * nu**2 * time_to_expiry
    return term1 * _z_over_x(z, rho) * inner


def _x_rho(z: float, rho: float) -> float:
    """Hagan chi(z) function; stable for z near 0."""
    # As z -> 0, chi(z) ~ z. Returning a constant here incorrectly forces
    # z/chi(z) -> 0 and collapses ATM vols toward zero.
    if abs(z) < 1e-10:
        return z
    return math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))


def _z_over_x(z: float, rho: float) -> float:
    """Stable ratio z / chi(z) with correct limit 1 at z=0."""
    if abs(z) < 1e-10:
        return 1.0
    return z / _x_rho(z, rho)
