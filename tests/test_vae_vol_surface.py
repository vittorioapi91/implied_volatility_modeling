"""Optional tests for VAE vol surface (requires torch)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from helper_module.vae_vol_surface import (
    N_GRID,
    TrainConfig,
    make_synthetic_sabr_surfaces,
    make_synthetic_ssvi_surfaces,
    train_vae,
)


def test_train_small() -> None:
    rng = np.random.default_rng(42)
    data = make_synthetic_sabr_surfaces(128, rng=rng)
    assert data.shape == (128, N_GRID)
    cfg = TrainConfig(epochs=2, batch_size=32, latent_dim=6)
    model, losses = train_vae(data, cfg)
    assert len(losses) == 2
    assert losses[-1] <= losses[0] * 1.5  # loose; may fluctuate slightly


def test_ssvi_generator_shape_and_finite() -> None:
    rng = np.random.default_rng(7)
    data = make_synthetic_ssvi_surfaces(64, rng=rng)
    assert data.shape == (64, N_GRID)
    assert np.all(np.isfinite(data))
    assert np.all(data > 0.0)
