# Implied volatility modeling

This repository explores **interest-rate implied volatility**: classical pricing primitives (Black–76, SABR), **discrete no-arbitrage diagnostics** on grids of calls or implied vols, optional **projection / repair** toward feasible surfaces, and **variational autoencoder** experiments that generate or regularize **volatility surfaces** on a fixed **tenor × delta** grid.

## Notebooks

- **`volatility_surfaces.ipynb`** — Visual surfaces for Black–76 (flat vol), SABR smiles, and a piecewise caplet-style slice (LMM-flavored, flat in strike).
- **`arbitrage_repair.ipynb`** — Intuition and demos for repair vs validation.
- **`vae_volatility_surface.ipynb`** — Main VAE workflow: baseline vs arbitrage-aware training, imputation, Monte Carlo-style diagnostics.
- **`vae_cnn_volatility_surface.ipynb`** — Convolutional encoder variant on the same grid and mixed SABR+SSVI setup.

**CLI:** after `pip install -e ".[vae]"`, run `python -m helper_module.vae_surface` (same flags as `main()` in `vae_vol_surface`) for a quick CPU demo.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"       # editable install + pytest
```

Optional extras:

- **`[notebook]`** — Jupyter.
- **`[vae]`** — PyTorch (required for VAE notebooks and related tests).

Example:

```bash
pip install -e ".[notebook,vae]"
```

Use the repo root (or `pip install -e .`) so `import helper_module` resolves from notebooks regardless of the kernel’s working directory.

## Tests

```bash
pytest tests/
```

Some tests skip when PyTorch is not installed; install **`[vae]`** to run the full set.

## Requirements

- Python **≥ 3.10**
- Core stack: NumPy, SciPy, Matplotlib; VAE code paths need **PyTorch** (see **`[vae]`**).
