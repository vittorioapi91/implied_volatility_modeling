"""Constrained-strike VAE decoder: strike-slice no-arb by construction."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from helper_module.arbitrage import validate_vol_surface_per_expiry_black76
from helper_module.vae_vol_surface import (
    DELTAS,
    ArbitrageAwareConfig,
    TrainConfig,
    TENORS_YEARS,
    bergeron_smoothness_penalty_torch,
    make_synthetic_sabr_surfaces,
    repair_surfaces_bergeron,
    strikes_for_bergeron_grid,
    train_vae_arbitrage_aware,
)


def test_constrained_decoder_zero_strike_violations_after_training() -> None:
    rng = np.random.default_rng(0)
    s = make_synthetic_sabr_surfaces(64, rng=rng)
    cfg = TrainConfig(epochs=2, batch_size=16, latent_dim=8)
    arb = ArbitrageAwareConfig(
        constrained_strike_decoder=True,
        lambda_arb=4.0,
        prior_arb_weight=1.0,
    )
    model, _ = train_vae_arbitrage_aware(s, cfg, arb)
    model.eval()
    dev = next(model.parameters()).device
    K = strikes_for_bergeron_grid(0.03)
    with torch.no_grad():
        z = torch.randn(6, cfg.latent_dim, device=dev)
        out = model.decode(z).cpu().numpy()
    for i in range(6):
        mat = out[i].reshape(len(TENORS_YEARS), len(DELTAS))
        rep = validate_vol_surface_per_expiry_black76(K, TENORS_YEARS, mat, 0.03)
        strike_msgs = [m for m in rep.violations if "Sticky-delta" not in m]
        assert not strike_msgs, strike_msgs[:3]


def test_strict_repair_targets_keep_only_passing_surfaces() -> None:
    rng = np.random.default_rng(0)
    s = make_synthetic_sabr_surfaces(10, rng=rng)
    repaired, info = repair_surfaces_bergeron(
        s,
        forward=0.03,
        max_iter=10,
        strict=True,
        return_details=True,
    )
    assert int(info["n_kept"]) == repaired.shape[0]
    assert repaired.shape[0] > 0
    K = strikes_for_bergeron_grid(0.03)
    for i in range(repaired.shape[0]):
        mat = repaired[i].reshape(len(TENORS_YEARS), len(DELTAS))
        rep = validate_vol_surface_per_expiry_black76(K, TENORS_YEARS, mat, 0.03)
        assert rep.ok


def test_smoothness_penalty_zero_for_affine_grid() -> None:
    # affine in tenor/delta => second differences are zero
    n_t = len(TENORS_YEARS)
    n_d = len(DELTAS)
    t = np.arange(n_t).reshape(n_t, 1)
    d = np.arange(n_d).reshape(1, n_d)
    mat = 0.2 + 0.01 * t + 0.02 * d
    recon = torch.tensor(mat.reshape(1, -1), dtype=torch.float32)
    p = bergeron_smoothness_penalty_torch(recon)
    assert float(p) < 1e-12


