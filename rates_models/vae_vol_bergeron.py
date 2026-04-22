"""
Variational autoencoder for implied volatility surfaces, following the setup in

  Bergeron, Fung, Hull, Poulos — "Variational Autoencoders: A Hands-Off Approach
  to Volatility" (arXiv:2102.03945).

**Grid (paper):** 8 tenors × 5 deltas = 40 points. Coordinates (log-tenor, delta) are
passed into a **pointwise decoder** (two hidden layers of 32 units). The encoder uses
two hidden layers of 32 units mapping the 40-D vol vector to Gaussian latent parameters.

Training uses the standard ELBO (reconstruction + KL). **Synthetic** SABR surfaces are
used by default so the module runs without proprietary FX data; replace
``make_synthetic_sabr_surfaces`` with historical surfaces for production.

**Imputation (paper step 2):** given observed entries on the grid, optimize the latent
vector z to minimize MSE on observed points (decoder fixed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rates_models.sabr import sabr_implied_vol_lognormal

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


# Paper-style FX grid: eight tenors (years), five deltas (call)
TENORS_YEARS = np.array(
    [1 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 3.0], dtype=np.float64
)
DELTAS = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)
N_GRID = len(TENORS_YEARS) * len(DELTAS)


def _require_torch():
    try:
        import torch
        import torch.nn as nn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "rates_models.vae_vol_bergeron requires PyTorch. Install with: pip install 'torch>=2.0'"
        ) from e
    return torch


def build_coordinate_features() -> np.ndarray:
    """
    Shape (40, 2): [log(1+T), delta] for each grid point (tenor-major, then delta).
    """
    rows = []
    for T in TENORS_YEARS:
        for d in DELTAS:
            rows.append([math.log1p(T), float(d)])
    return np.asarray(rows, dtype=np.float64)


def _rough_strike_from_delta(forward: float, T: float, delta: float) -> float:
    """Map delta to a positive strike for SABR demo (not market-standard quoting)."""
    z = max(min(delta, 1.0 - 1e-6), 1e-6)
    log_m = 0.12 * (z - 0.5) + 0.03 * math.sin(2.5 * T)
    return max(forward * math.exp(log_m), 1e-8)


def strikes_for_bergeron_grid(forward: float) -> np.ndarray:
    """
    Black-76 strikes ``K[i, j]`` at tenor ``i`` and delta ``j``, using the same
    ``_rough_strike_from_delta`` mapping as :func:`make_synthetic_sabr_surfaces`.
    Use with a vol matrix of shape ``(len(TENORS_YEARS), len(DELTAS))`` when running
    discrete no-arbitrage checks (e.g. :func:`rates_models.arbitrage.validate_vol_surface`).
    """
    out = np.empty((len(TENORS_YEARS), len(DELTAS)), dtype=np.float64)
    for i, T in enumerate(TENORS_YEARS):
        for j, d in enumerate(DELTAS):
            out[i, j] = _rough_strike_from_delta(forward, float(T), float(d))
    return out


def make_synthetic_sabr_surfaces(
    n_samples: int,
    rng: np.random.Generator | None = None,
    forward: float = 0.03,
) -> np.ndarray:
    """
    Random SABR parameters → 40-point lognormal implied vol surfaces (n_samples, 40).
    """
    rng = rng or np.random.default_rng()
    out = np.zeros((n_samples, N_GRID), dtype=np.float64)
    for s in range(n_samples):
        alpha = float(rng.uniform(0.008, 0.035))
        beta = float(rng.uniform(0.15, 0.85))
        rho = float(rng.uniform(-0.6, 0.6))
        nu = float(rng.uniform(0.15, 0.55))
        idx = 0
        for T in TENORS_YEARS:
            for d in DELTAS:
                K = _rough_strike_from_delta(forward, T, d)
                out[s, idx] = sabr_implied_vol_lognormal(
                    forward, K, float(T), alpha, beta, rho, nu
                )
                idx += 1
    return out


@dataclass
class TrainConfig:
    latent_dim: int = 8
    hidden: int = 32
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    kl_weight: float = 1.0


def black76_call_torch(
    F: "torch.Tensor",
    K: "torch.Tensor",
    T: "torch.Tensor",
    sigma: "torch.Tensor",
    discount: "torch.Tensor",
) -> "torch.Tensor":
    """Black-76 call on forward; all tensors broadcast-compatible."""
    torch = _require_torch()
    eps = 1e-10
    T = torch.clamp(T, min=eps)
    sig = torch.clamp(sigma, min=eps)
    sqrt_t = torch.sqrt(T)
    d1 = (torch.log(F / K) + 0.5 * sig**2 * T) / (sig * sqrt_t + eps)
    d2 = d1 - sig * sqrt_t
    n = torch.distributions.Normal(
        torch.zeros((), device=sigma.device, dtype=sigma.dtype),
        torch.ones((), device=sigma.device, dtype=sigma.dtype),
    )
    return discount * (F * n.cdf(d1) - K * n.cdf(d2))


def black76_implied_vol_newton_torch(
    target_price: "torch.Tensor",
    F: "torch.Tensor",
    K: "torch.Tensor",
    T: "torch.Tensor",
    discount: "torch.Tensor",
    *,
    n_iter: int = 56,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 6.0,
) -> "torch.Tensor":
    """
    Batched implied σ from Black-76 price via monotone bisection (vega > 0).
    Float64 interior for bracket accuracy; short tenors handled better than Newton.
    """
    torch = _require_torch()
    out_dtype = target_price.dtype
    eps = 1e-14
    F64 = F.to(torch.float64)
    K64 = K.to(torch.float64)
    T64 = torch.clamp(T.to(torch.float64), min=1e-12)
    discount64 = discount.to(torch.float64)
    price64 = target_price.to(torch.float64)
    intrinsic = torch.clamp(F64 - K64, min=0.0) * discount64
    upper = F64 * discount64
    price = torch.clamp(price64, min=intrinsic + 1e-15, max=upper - 1e-15)

    lo = torch.full_like(price, sigma_lo, dtype=torch.float64)
    hi = torch.full_like(price, float(sigma_hi), dtype=torch.float64)
    # Ensure Black(hi) >= price everywhere (expand upper bracket if needed).
    for _ in range(14):
        p_hi = black76_call_torch(F64, K64, T64, hi, discount64)
        hi = torch.where(
            p_hi < price,
            torch.clamp(hi * 1.65, max=40.0),
            hi,
        )

    for _ in range(n_iter):
        mid = (lo + hi) * 0.5
        p = black76_call_torch(
            F64, K64, T64, mid, discount64
        )
        # Black price increases in σ; if p > target, shrink σ
        too_high = p > price
        hi = torch.where(too_high, mid, hi)
        lo = torch.where(too_high, lo, mid)
    sigma = (lo + hi) * 0.5
    return sigma.to(out_dtype)


def call_prices_monotone_convex_five_strikes(
    raw: "torch.Tensor",
    *,
    price_scale: float = 0.04,
    forward: float = 0.03,
    discount: float = 1.0,
) -> "torch.Tensor":
    """
    Map 5 unconstrained logits to call prices at five strikes (ascending K) that are
    strictly decreasing and discretely convex (non-negative second difference).

    ``raw`` shape (..., 5). Returns ``C`` shape (..., 5) with C[...,0] largest (lowest
    strike). Prices are scaled so ``max C <= forward * discount`` so Black-76 inversion
    is well-posed.
    """
    torch = _require_torch()
    s_base = price_scale * 0.25
    s_gap = price_scale * 0.35
    g3 = torch.nn.functional.softplus(raw[..., 0]) * s_gap
    g2 = g3 + torch.nn.functional.softplus(raw[..., 1]) * s_gap
    g1 = g2 + torch.nn.functional.softplus(raw[..., 2]) * s_gap
    g0 = g1 + torch.nn.functional.softplus(raw[..., 3]) * s_gap
    c4 = torch.nn.functional.softplus(raw[..., 4]) * s_base
    c3 = c4 + g3
    c2 = c3 + g2
    c1 = c2 + g1
    c0 = c1 + g0
    c = torch.stack([c0, c1, c2, c3, c4], dim=-1)
    cap = raw.new_tensor(forward * discount * 0.999)
    peak = c[..., 0:1]
    c = c * torch.minimum(torch.ones_like(peak), cap / torch.clamp(peak, min=1e-12))
    return c


@dataclass
class ArbitrageAwareConfig:
    """
    **Training-data filter (encoder side):** When ``repair_targets`` is True, each
    synthetic surface is projected onto the discrete no-arbitrage set (Bergeron grid)
    before ELBO, so the VAE does not fit raw arbitrage-violating targets.

    **Decoder regularizer:** ``lambda_arb`` scales soft Black–76 violation terms. They
    are usually **much smaller than MSE**, so use ``butterfly_weight`` /
    ``prior_arb_weight`` and raise ``lambda_arb`` until the arb term is a visible
    fraction of total loss. The prior penalty trains ``decode(z)`` for ``z \\sim
    \\mathcal{N}(0,I)``, matching **sampling** at inference (reconstruction-only
    penalties often ignore that path). None of this **guarantees** a feasible surface;
    for a hard guarantee on the grid, apply :func:`rates_models.arbitrage_repair.repair_vol_bergeron_grid_black76` after decoding.

    **Constrained decoder:** If ``constrained_strike_decoder`` is True, the decoder outputs
    (per expiry) five logits that are turned into **call prices** that satisfy **strike**
    monotonicity and **discrete butterfly** by construction, then mapped to σ with
    differentiable Newton. **Calendar** (sticky total variance) is **not** automatic and
    is still trained via the sticky term in :func:`bergeron_arbitrage_penalty_torch` (set
    mono/butterfly weights to 0 when using this decoder).

    **Performance:** Repairing every training surface is expensive (SciPy projections).
    Keep ``repair_targets=False`` for large ``n_train``, or precompute repaired data
    offline; use ``repair_targets=True`` mainly for small datasets or demos.
    """

    forward: float = 0.03
    lambda_arb: float = 4.0
    mono_weight: float = 1.0
    butterfly_weight: float = 12.0
    sticky_weight: float = 1.0
    #: Extra decode(z), z~N(0,I), penalty per batch (matches generative sampling).
    prior_arb_weight: float = 1.0
    #: Smoothness penalty on decoded vol grids (discrete second derivatives).
    smoothness_weight: float = 0.0
    smoothness_tenor_weight: float = 1.0
    smoothness_delta_weight: float = 1.0
    repair_targets: bool = False
    repair_max_iter: int = 12
    #: Retry multipliers for repair iterations (e.g. 12, 24, 48).
    repair_retry_scales: tuple[int, ...] = (1, 2, 4)
    #: Keep only repaired targets that pass discrete checker.
    strict_repair_targets: bool = True
    #: Raise if strict filtering leaves too little calibration data.
    min_repaired_fraction: float = 0.9
    #: If True, decoder never outputs σ freely; strike slice no-arb is structural.
    constrained_strike_decoder: bool = False
    #: Optional built-in no-arb projection layer inside decoder forward pass.
    enforce_full_noarb_layer: bool = True
    full_noarb_max_iter: int = 20


def _build_module(torch, config: TrainConfig):
    nn = torch.nn

    class VolSurfaceVAE(nn.Module):
        """
        Pointwise decoder: for each of 40 locations, MLP([z, u]) -> σ_imp.
        Encoder: MLP(x) -> mu, logvar.
        """

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "coord", torch.tensor(build_coordinate_features(), dtype=torch.float32)
            )
            L = config.latent_dim
            h = config.hidden
            self.encoder = nn.Sequential(
                nn.Linear(N_GRID, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 2 * L),
            )
            in_dec = L + 2
            self.dec_fc1 = nn.Linear(in_dec, h)
            self.dec_fc2 = nn.Linear(h, h)
            self.dec_out = nn.Linear(h, 1)

        def encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = self.encoder(x)
            L = h.shape[-1] // 2
            return h[:, :L], h[:, L:]

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            """Batch z: (B, L) -> vol grid (B, 40)."""
            B = z.shape[0]
            coord = self.coord.unsqueeze(0).expand(B, -1, -1)
            z_exp = z.unsqueeze(1).expand(B, N_GRID, -1)
            h = torch.cat([z_exp, coord], dim=-1)
            h = h.reshape(B * N_GRID, -1)
            h = torch.relu(self.dec_fc1(h))
            h = torch.relu(self.dec_fc2(h))
            o = self.dec_out(h).squeeze(-1)
            return o.reshape(B, N_GRID)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            recon = self.decode(z)
            return recon, mu, logvar

    return VolSurfaceVAE()


def _build_module_constrained_strike(
    torch,
    config: TrainConfig,
    forward: float,
    *,
    enforce_full_noarb_layer: bool,
    full_noarb_max_iter: int,
):
    """Decoder outputs monotone convex call prices per expiry → σ via Newton (strike no-arb by construction)."""
    nn = torch.nn
    n_t, n_d = len(TENORS_YEARS), len(DELTAS)
    assert n_d == 5

    K_np = strikes_for_bergeron_grid(forward)

    class VolSurfaceVAEConstrainedStrike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "K_grid", torch.tensor(K_np, dtype=torch.float32).view(n_t, n_d)
            )
            self.register_buffer(
                "T_row", torch.tensor(TENORS_YEARS, dtype=torch.float32).view(n_t, 1)
            )
            L = config.latent_dim
            h = config.hidden
            self.encoder = nn.Sequential(
                nn.Linear(N_GRID, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 2 * L),
            )
            self.row_net = nn.Sequential(
                nn.Linear(L + 1, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 5),
            )

        def encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = self.encoder(x)
            L = h.shape[-1] // 2
            return h[:, :L], h[:, L:]

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            B = z.shape[0]
            n_t = self.K_grid.shape[0]
            log_t = torch.log1p(self.T_row).view(n_t)  # (n_t,)
            F = z.new_tensor(forward)
            D = torch.ones((), device=z.device, dtype=z.dtype)
            rows: list[torch.Tensor] = []
            for i in range(n_t):
                inp = torch.cat([z, log_t[i].expand(B, 1)], dim=-1)
                raw = self.row_net(inp)
                C_row = call_prices_monotone_convex_five_strikes(
                    raw, forward=forward, discount=1.0
                )
                K_row = self.K_grid[i : i + 1, :].expand(B, -1)
                T_row = self.T_row[i : i + 1, :].expand(B, -1)
                sig_row = black76_implied_vol_newton_torch(
                    C_row, F, K_row, T_row, D
                )
                rows.append(sig_row)
            out = torch.stack(rows, dim=1)
            if enforce_full_noarb_layer and (not self.training):
                # Straight-through no-arb projection layer: forward uses projected vols,
                # backward keeps gradients from the pre-projection decoder output.
                out = _project_bergeron_noarb_layer_torch(
                    out, forward=forward, max_iter=full_noarb_max_iter
                )
            return out.reshape(B, N_GRID)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            recon = self.decode(z)
            return recon, mu, logvar

    return VolSurfaceVAEConstrainedStrike()


def loss_elbo(
    recon: "torch.Tensor",
    target: "torch.Tensor",
    mu: "torch.Tensor",
    logvar: "torch.Tensor",
    kl_weight: float = 1.0,
) -> "torch.Tensor":
    torch = _require_torch()
    mse = torch.nn.functional.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl_weight * kl


def train_vae(
    surfaces: np.ndarray,
    config: TrainConfig | None = None,
    device: str | None = None,
) -> tuple[object, list[float]]:
    """
    Train VAE on array (n_samples, 40). Returns (model, losses per epoch).
    """
    torch = _require_torch()
    config = config or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    x = torch.tensor(surfaces, dtype=torch.float32, device=dev)
    model = _build_module(torch, config).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    n = x.shape[0]
    losses: list[float] = []

    model.train()
    for epoch in range(config.epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch = x[idx]
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss = loss_elbo(recon, batch, mu, logvar, config.kl_weight)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))

    return model, losses


def repair_surfaces_bergeron(
    surfaces: np.ndarray,
    forward: float,
    discount: float = 1.0,
    *,
    max_iter: int = 12,
    strict: bool = True,
    retry_scales: tuple[int, ...] = (1, 2, 4),
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """
    Project each training surface (n, 40) onto the discrete no-arbitrage set used in
    :func:`rates_models.arbitrage_repair.repair_vol_bergeron_grid_black76` (same strike
    grid as synthetic SABR).

    Imported lazily so the rest of this module loads even if an older ``arbitrage_repair``
    is cached until a notebook reloads it.
    """
    from rates_models.arbitrage_repair import repair_vol_bergeron_grid_black76
    from rates_models.arbitrage import validate_vol_surface_per_expiry_black76

    K = strikes_for_bergeron_grid(forward)
    n_t, n_d = len(TENORS_YEARS), len(DELTAS)
    out_rows: list[np.ndarray] = []
    kept_idx: list[int] = []
    dropped_idx: list[int] = []
    retry_scales = tuple(int(s) for s in retry_scales if int(s) > 0) or (1,)
    for i in range(surfaces.shape[0]):
        mat = surfaces[i].reshape(n_t, n_d)
        pre = validate_vol_surface_per_expiry_black76(
            K, TENORS_YEARS, mat, forward, discount
        )
        if pre.ok:
            out_rows.append(mat.ravel())
            kept_idx.append(i)
            continue
        last_mat = mat
        ok = False
        for scale in retry_scales:
            tries = int(max_iter * scale)
            mat_r, rep = repair_vol_bergeron_grid_black76(
                last_mat,
                strikes=K,
                tenors=TENORS_YEARS,
                forward=forward,
                discount=discount,
                max_iter=tries,
            )
            last_mat = mat_r
            if rep.ok:
                ok = True
                break
        if ok or not strict:
            out_rows.append(last_mat.ravel())
            kept_idx.append(i)
        else:
            dropped_idx.append(i)
    if out_rows:
        out = np.vstack(out_rows).astype(np.float64, copy=False)
    else:
        out = np.empty((0, n_t * n_d), dtype=np.float64)
    if return_details:
        return out, {
            "n_input": int(surfaces.shape[0]),
            "n_kept": int(len(kept_idx)),
            "n_dropped": int(len(dropped_idx)),
            "kept_idx": np.array(kept_idx, dtype=int),
            "dropped_idx": np.array(dropped_idx, dtype=int),
            "strict": bool(strict),
            "retry_scales": retry_scales,
            "base_max_iter": int(max_iter),
        }
    return out


def _project_bergeron_noarb_layer_torch(
    sigma: "torch.Tensor",
    *,
    forward: float,
    max_iter: int,
) -> "torch.Tensor":
    """
    Project each (tenor, delta) surface in a batch to the Bergeron-grid no-arb set.
    Forward output is projected; gradient is straight-through identity.
    """
    from rates_models.arbitrage_repair import repair_vol_bergeron_grid_black76

    n_t, n_d = len(TENORS_YEARS), len(DELTAS)
    if sigma.ndim != 3 or sigma.shape[1:] != (n_t, n_d):
        raise ValueError("sigma must have shape (batch, n_tenors, n_deltas).")
    K = strikes_for_bergeron_grid(forward)
    arr = sigma.detach().cpu().numpy()
    out = np.empty_like(arr)
    for b in range(arr.shape[0]):
        mat_r, _ = repair_vol_bergeron_grid_black76(
            arr[b],
            strikes=K,
            tenors=TENORS_YEARS,
            forward=forward,
            max_iter=max_iter,
        )
        # Numerically enforce non-decreasing sticky-delta total variance with tiny margin.
        tcol = TENORS_YEARS.reshape(-1, 1)
        w = (mat_r**2) * tcol
        w = np.maximum.accumulate(
            w + (1e-5 * np.arange(tcol.shape[0]).reshape(-1, 1)),
            axis=0,
        )
        mat_r = np.sqrt(np.maximum(w / tcol, 0.0))
        out[b] = mat_r
    proj = _require_torch().tensor(out, dtype=sigma.dtype, device=sigma.device)
    return sigma + (proj - sigma).detach()


def bergeron_arbitrage_penalty_torch(
    recon: "torch.Tensor",
    K_grid: "torch.Tensor",
    T_years: "torch.Tensor",
    forward: float,
    discount: float = 1.0,
    *,
    mono_weight: float = 1.0,
    butterfly_weight: float = 1.0,
    sticky_weight: float = 1.0,
) -> "torch.Tensor":
    """
    Differentiable soft violation measure for Black–76 calls on the Bergeron grid:
    strike monotonicity, discrete butterfly, sticky-delta total variance along tenor.
    ``recon`` is (batch, 40); ``K_grid`` and ``T_years`` match ``strikes_for_bergeron_grid``
    and ``TENORS_YEARS`` on the same device/dtype as ``recon``.

    Squared violations emphasize butterfly breaches (the usual discrete-check failure)
    with larger gradients when violations are small.
    """
    torch = _require_torch()
    n_t = T_years.shape[0]
    n_d = K_grid.shape[1]
    B = recon.shape[0]
    s = recon.view(B, n_t, n_d)
    F = recon.new_tensor(forward)
    D = recon.new_tensor(discount)
    Kb = K_grid.unsqueeze(0).expand(B, -1, -1)
    T_exp = T_years.view(1, n_t, 1).expand(B, -1, n_d)
    eps = 1e-6
    sqrt_t = torch.sqrt(torch.clamp(T_exp, min=eps))
    sig = torch.clamp(s, min=eps)
    d1 = (torch.log(F / Kb) + 0.5 * sig**2 * T_exp) / (sig * sqrt_t + eps)
    normal = torch.distributions.Normal(
        torch.zeros((), device=recon.device, dtype=recon.dtype),
        torch.ones((), device=recon.device, dtype=recon.dtype),
    )
    C = D * (F * normal.cdf(d1) - Kb * normal.cdf(d1 - sig * sqrt_t))
    v_mono = torch.relu(C[..., 1:] - C[..., :-1])
    mono = mono_weight * v_mono.pow(2).mean()
    if n_d >= 3:
        v_bf = torch.relu(
            -(C[..., :-2] - 2.0 * C[..., 1:-1] + C[..., 2:])
        )
        butterfly = butterfly_weight * v_bf.pow(2).mean()
    else:
        butterfly = recon.new_zeros(())
    w = s**2 * T_exp
    if n_t >= 2:
        v_st = torch.relu(w[:, :-1, :] - w[:, 1:, :])
        sticky = sticky_weight * v_st.pow(2).mean()
    else:
        sticky = recon.new_zeros(())
    return mono + butterfly + sticky


def bergeron_smoothness_penalty_torch(
    recon: "torch.Tensor",
    *,
    tenor_weight: float = 1.0,
    delta_weight: float = 1.0,
) -> "torch.Tensor":
    """
    Discrete smoothness proxy on the 8x5 grid using squared second differences.

    This does not mathematically guarantee a global C2 surface for all interpolated
    points, but it strongly suppresses curvature kinks and high-frequency oscillations
    in both tenor and delta directions.
    """
    torch = _require_torch()
    B = recon.shape[0]
    s = recon.view(B, len(TENORS_YEARS), len(DELTAS))
    penalty = recon.new_zeros(())
    if s.shape[1] >= 3:
        d2_t = s[:, 2:, :] - 2.0 * s[:, 1:-1, :] + s[:, :-2, :]
        penalty = penalty + tenor_weight * d2_t.pow(2).mean()
    if s.shape[2] >= 3:
        d2_d = s[:, :, 2:] - 2.0 * s[:, :, 1:-1] + s[:, :, :-2]
        penalty = penalty + delta_weight * d2_d.pow(2).mean()
    return penalty


def train_vae_arbitrage_aware(
    surfaces: np.ndarray,
    config: TrainConfig | None = None,
    arb_config: ArbitrageAwareConfig | None = None,
    device: str | None = None,
    *,
    return_components: bool = False,
) -> tuple[object, list[float]] | tuple[object, list[float], dict[str, list[float]]]:
    """
    Train the same :class:`VolSurfaceVAE` with (optional) repaired targets and an
    arbitrage penalty on decoder outputs; see :class:`ArbitrageAwareConfig`.
    """
    torch = _require_torch()
    config = config or TrainConfig()
    arb_config = arb_config or ArbitrageAwareConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)

    if arb_config.repair_targets:
        x_np, rep_info = repair_surfaces_bergeron(
            surfaces,
            arb_config.forward,
            max_iter=arb_config.repair_max_iter,
            strict=arb_config.strict_repair_targets,
            retry_scales=arb_config.repair_retry_scales,
            return_details=True,
        )
        kept_fraction = (
            rep_info["n_kept"] / max(rep_info["n_input"], 1)  # type: ignore[index]
        )
        if rep_info["n_kept"] == 0:  # type: ignore[index]
            raise RuntimeError(
                "repair_targets produced zero feasible targets; increase "
                "repair_max_iter/retry_scales or disable strict_repair_targets."
            )
        if (
            arb_config.strict_repair_targets
            and kept_fraction < arb_config.min_repaired_fraction
        ):
            raise RuntimeError(
                "Strict repair kept too few targets: "
                f"{rep_info['n_kept']}/{rep_info['n_input']} "
                f"({kept_fraction:.1%}) < min_repaired_fraction="
                f"{arb_config.min_repaired_fraction:.1%}. "
                "Increase repair_max_iter/retry_scales, lower min_repaired_fraction, "
                "or disable strict_repair_targets."
            )
        print(
            "repair_targets summary:",
            f"kept {rep_info['n_kept']}/{rep_info['n_input']}",
            f"({kept_fraction:.1%}), dropped {rep_info['n_dropped']}",
            f"(strict={arb_config.strict_repair_targets})",
        )
    else:
        x_np = np.asarray(surfaces, dtype=np.float64)
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)

    K_grid = torch.tensor(
        strikes_for_bergeron_grid(arb_config.forward), dtype=torch.float32, device=dev
    )
    T_years = torch.tensor(TENORS_YEARS, dtype=torch.float32, device=dev)

    if arb_config.constrained_strike_decoder:
        model = _build_module_constrained_strike(
            torch,
            config,
            arb_config.forward,
            enforce_full_noarb_layer=arb_config.enforce_full_noarb_layer,
            full_noarb_max_iter=arb_config.full_noarb_max_iter,
        ).to(dev)
        mono_w = 0.0
        bf_w = 0.0
        sticky_w = arb_config.sticky_weight
    else:
        model = _build_module(torch, config).to(dev)
        mono_w = arb_config.mono_weight
        bf_w = arb_config.butterfly_weight
        sticky_w = arb_config.sticky_weight

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    n = x.shape[0]
    losses: list[float] = []
    hist_elbo: list[float] = []
    hist_arb: list[float] = []
    hist_smooth: list[float] = []

    model.train()
    for epoch in range(config.epochs):
        perm = torch.randperm(n, device=dev)
        epoch_loss = 0.0
        epoch_elbo = 0.0
        epoch_arb = 0.0
        epoch_smooth = 0.0
        steps = 0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch = x[idx]
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            elbo = loss_elbo(recon, batch, mu, logvar, config.kl_weight)
            arb = bergeron_arbitrage_penalty_torch(
                recon,
                K_grid,
                T_years,
                arb_config.forward,
                mono_weight=mono_w,
                butterfly_weight=bf_w,
                sticky_weight=sticky_w,
            )
            if arb_config.prior_arb_weight > 0.0:
                z_p = torch.randn(
                    batch.shape[0], config.latent_dim, device=dev, dtype=batch.dtype
                )
                dec_p = model.decode(z_p)
                arb_p = bergeron_arbitrage_penalty_torch(
                    dec_p,
                    K_grid,
                    T_years,
                    arb_config.forward,
                    mono_weight=mono_w,
                    butterfly_weight=bf_w,
                    sticky_weight=sticky_w,
                )
                arb = arb + arb_config.prior_arb_weight * arb_p
            smooth = bergeron_smoothness_penalty_torch(
                recon,
                tenor_weight=arb_config.smoothness_tenor_weight,
                delta_weight=arb_config.smoothness_delta_weight,
            )
            loss = elbo + arb_config.lambda_arb * arb + arb_config.smoothness_weight * smooth
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            epoch_elbo += float(elbo.item())
            epoch_arb += float(arb.item())
            epoch_smooth += float(smooth.item())
            steps += 1
        denom = max(steps, 1)
        losses.append(epoch_loss / denom)
        hist_elbo.append(epoch_elbo / denom)
        hist_arb.append(epoch_arb / denom)
        hist_smooth.append(epoch_smooth / denom)

    if return_components:
        return model, losses, {
            "elbo": hist_elbo,
            "arb_raw": hist_arb,
            "smooth_raw": hist_smooth,
        }
    return model, losses


def encode_mean_surface(
    model: object,
    surface: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """Deterministic latent mean for a single surface (1, 40)."""
    torch = _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    x = torch.tensor(surface.reshape(1, -1), dtype=torch.float32, device=dev)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.cpu().numpy().squeeze(0)


def impute_surface_latent_search(
    model: object,
    observed_mask: np.ndarray,
    observed_vols: np.ndarray,
    config: TrainConfig | None = None,
    steps: int = 400,
    lr: float = 0.05,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find latent z minimizing MSE on observed grid points, then decode full surface.

    Parameters
    ----------
    observed_mask : bool array (40,)
        True where σ is observed.
    observed_vols : float array (40,)
        Values (ignored where mask is False; can use 0 there).

    Returns
    -------
    z : (latent_dim,)
    surface_full : (40,)
    """
    torch = _require_torch()
    config = config or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    mask = torch.tensor(observed_mask, dtype=torch.bool, device=dev)
    target = torch.tensor(observed_vols, dtype=torch.float32, device=dev)

    z = torch.zeros(1, config.latent_dim, device=dev, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        recon = model.decode(z)
        err = (recon[0, mask] - target[mask]).pow(2).mean()
        err.backward()
        opt.step()

    with torch.no_grad():
        full = model.decode(z).cpu().numpy().squeeze(0)
    return z.detach().cpu().numpy().squeeze(0), full


def main() -> None:
    """Quick CPU demo: train on synthetic SABR, impute from partial observations."""
    import argparse

    p = argparse.ArgumentParser(description="Train Bergeron et al. (2021) vol VAE demo.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = TrainConfig(epochs=args.epochs, latent_dim=args.latent)
    data = make_synthetic_sabr_surfaces(args.n_samples, rng=rng)
    model, losses = train_vae(data, cfg)
    print("Final epoch loss:", losses[-1])
    # Impute: hide half the grid
    truth = data[0]
    mask = rng.random(N_GRID) > 0.5
    z_hat, filled = impute_surface_latent_search(model, mask, truth, config=cfg)
    err = np.sqrt(np.mean((filled[~mask] - truth[~mask]) ** 2))
    print("RMSE on hidden points (demo):", err)


if __name__ == "__main__":
    main()
