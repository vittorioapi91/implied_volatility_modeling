"""Black (1976) on forward rates + Bachelier (normal) for completeness."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import optimize, stats


def _phi(x: float | np.ndarray) -> float | np.ndarray:
    return stats.norm.cdf(x)


def black76_price(
    forward: float,
    strike: float,
    time_to_expiry: float,
    sigma: float,
    discount: float = 1.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black-76 price for an option on a forward rate (e.g. caplet / swaption core).

    P = D * (F * N(d1) - K * N(d2)) for a call on the forward.
    """
    if time_to_expiry <= 0.0:
        if option_type == "call":
            return discount * max(forward - strike, 0.0)
        return discount * max(strike - forward, 0.0)
    if sigma <= 0.0:
        if option_type == "call":
            return discount * max(forward - strike, 0.0)
        return discount * max(strike - forward, 0.0)

    sqrt_t = math.sqrt(time_to_expiry)
    if forward > 0.0 and strike > 0.0:
        d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * time_to_expiry) / (
            sigma * sqrt_t
        )
    else:
        d1 = float("inf") if forward > strike else float("-inf")
    d2 = d1 - sigma * sqrt_t
    if option_type == "call":
        return discount * (forward * _phi(d1) - strike * _phi(d2))
    return discount * (strike * _phi(-d2) - forward * _phi(-d1))


def black76_implied_vol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    market_price: float,
    discount: float = 1.0,
    option_type: Literal["call", "put"] = "call",
    *,
    bracket: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """Root-find Black implied volatility (annualized lognormal)."""

    def objective(sig: float) -> float:
        return (
            black76_price(
                forward, strike, time_to_expiry, sig, discount, option_type
            )
            - market_price
        )

    lo, hi = bracket
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            "Black implied vol: price not bracketed; adjust forward/strike/price or bracket."
        )
    return float(optimize.brentq(objective, lo, hi, xtol=1e-10, rtol=1e-10))


def bachelier_price(
    forward: float,
    strike: float,
    time_to_expiry: float,
    sigma_n: float,
    discount: float = 1.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Bachelier (normal) model: dF = sigma_n dW (sigma_n in rate units per sqrt year).

    Call = D * ((F-K)*N(d) + sigma_n*sqrt(T)*n(d)), d = (F-K)/(sigma_n*sqrt(T)).
    """
    if time_to_expiry <= 0.0:
        if option_type == "call":
            return discount * max(forward - strike, 0.0)
        return discount * max(strike - forward, 0.0)
    if sigma_n <= 0.0:
        if option_type == "call":
            return discount * max(forward - strike, 0.0)
        return discount * max(strike - forward, 0.0)

    sqrt_t = math.sqrt(time_to_expiry)
    d = (forward - strike) / (sigma_n * sqrt_t)
    pdf = stats.norm.pdf(d)
    if option_type == "call":
        return discount * (
            (forward - strike) * _phi(d) + sigma_n * sqrt_t * pdf
        )
    return discount * (
        (strike - forward) * _phi(-d) + sigma_n * sqrt_t * pdf
    )
