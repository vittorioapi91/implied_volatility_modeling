#!/usr/bin/env python3
"""
CLI entry for volatility-surface VAE demo.

Install: pip install -e ".[vae]"

Usage:
  python scripts/vae_surface.py --epochs 40 --n-samples 2000
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rates_models.vae_surface import main  # noqa: E402

if __name__ == "__main__":
    main()
