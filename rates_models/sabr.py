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
    fk = f * k
    log_fk = math.log(f / k)

    if abs(beta - 1.0) < 1e-12:
        # beta -> 1: lognormal SABR (CEV beta=1)
        term1 = alpha / (
            fk ** 0.25 * (1.0 + (1.0 / 24.0) * log_fk**2 + (1.0 / 1920.0) * log_fk**4)
        )
        z = nu / alpha * log_fk
        x_z = _x_rho(z, rho)
        inner = 1.0 + ((1.0 / 24.0) * alpha**2 + (1.0 / 4.0 - rho**2 / 12.0) * nu**2) * time_to_expiry
        return term1 * z / x_z * inner

    if abs(beta) < 1e-12:
        # beta = 0: normal SABR — caller may prefer sabr_normal_vol; this returns *Black* vol
        # Hagan normal expansion gives sigma_N; convert approximately via ATM: sigma_B ≈ sigma_N / F
        sigma_n = sabr_normal_vol(forward, strike, time_to_expiry, alpha, rho, nu)
        atm = max(0.5 * (f + k), eps)
        return sigma_n / atm

    fk_mid = (f + k) * 0.5
    term1_num = alpha * (1.0 + (1.0 - beta) ** 2 / 24.0 * log_fk**2 + (1.0 - beta) ** 4 / 1920.0 * log_fk**4)
    term1_den = fk_mid ** (1.0 - beta) * (
        1.0 + (1.0 - beta) ** 2 / 24.0 * (math.log(fk) ** 2) + (1.0 - beta) ** 4 / 1920.0 * (math.log(fk) ** 4)
    )
    term1 = term1_num / term1_den

    z = nu / alpha * (fk_mid ** (1.0 - beta)) * log_fk
    x_z = _x_rho(z, rho)

    inner = (
        1.0
        + (
            (1.0 - beta) ** 2 / 24.0 * alpha**2 / (fk_mid ** (2.0 - 2.0 * beta))
            + 0.25 * rho * beta * nu * alpha / (fk_mid ** (1.0 - beta))
            + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
        )
        * time_to_expiry
    )
    return term1 * z / x_z * inner


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
    x_z = _x_rho(z, rho)
    inner = 1.0 + (-(1.0 / 3.0) * rho**2 + 0.25) * nu**2 * time_to_expiry
    return term1 * z / x_z * inner


def _x_rho(z: float, rho: float) -> float:
    """Hagan chi(z) function; stable for z near 0."""
    if abs(z) < 1e-14:
        return 1.0 - rho
    return math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
