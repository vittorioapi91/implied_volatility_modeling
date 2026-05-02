"""Neutral aliases for the volatility-surface VAE utilities."""

from helper_module.vae_vol_surface import (
    DELTAS,
    TENORS_YEARS,
    ArbitrageAwareConfig,
    TrainConfig,
    bergeron_smoothness_penalty_torch as surface_smoothness_penalty_torch,
    impute_surface_latent_search,
    make_synthetic_sabr_surfaces,
    make_synthetic_ssvi_surfaces,
    main,
    pick_torch_training_device,
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
    "make_synthetic_ssvi_surfaces",
    "main",
    "pick_torch_training_device",
    "strikes_for_surface_grid",
    "train_vae",
    "train_vae_arbitrage_aware",
]


if __name__ == "__main__":
    main()
