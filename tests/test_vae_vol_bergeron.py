"""Optional tests for VAE vol surface (requires torch)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from rates_models.vae_vol_bergeron import (
    N_GRID,
    TrainConfig,
    make_synthetic_sabr_surfaces,
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
