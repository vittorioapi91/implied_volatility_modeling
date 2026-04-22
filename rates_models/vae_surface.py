"""Neutral aliases for the volatility-surface VAE utilities."""

from rates_models.vae_vol_bergeron import (
    DELTAS,
    TENORS_YEARS,
    ArbitrageAwareConfig,
    TrainConfig,
    bergeron_smoothness_penalty_torch as surface_smoothness_penalty_torch,
    impute_surface_latent_search,
    make_synthetic_sabr_surfaces,
    main,
    strikes_for_bergeron_grid as strikes_for_surface_grid,
    train_vae,
    train_vae_arbitrage_aware,
)

__all__ = [
    "DELTAS",
    "TENORS_YEARS",
    "ArbitrageAwareConfig",
    "TrainConfig",
    "surface_smoothness_penalty_torch",
    "impute_surface_latent_search",
    "make_synthetic_sabr_surfaces",
    "main",
    "strikes_for_surface_grid",
    "train_vae",
    "train_vae_arbitrage_aware",
]
