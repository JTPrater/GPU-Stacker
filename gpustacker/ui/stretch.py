# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Display stretches for previews (numpy only): PixInsight-style auto STF and helpers."""
from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826


def mtf(m: float, x):
    """Midtones transfer function."""
    x = np.asarray(x, dtype=np.float32)
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def _robust(v: np.ndarray) -> tuple[float, float]:
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * MAD_TO_SIGMA
    return med, mad


def auto_stf(data: np.ndarray, linked: bool = False, target: float = 0.25, clip: float = -2.8,
             sample: int = 250_000) -> list[tuple[float, float, float]]:
    """(shadows, midtones, highlights) per channel from the non-zero pixels of (C, h, w) data.

    Values are scaled by the 99.995th percentile first, so it works on linear data in any range."""
    c = data.shape[0]
    flat = data.reshape(c, -1)
    nz = np.any(flat != 0, axis=0)
    idx = np.flatnonzero(nz)
    if idx.size == 0:
        return [(0.0, 0.5, 1.0)] * c
    if idx.size > sample:
        idx = idx[np.linspace(0, idx.size - 1, sample).astype(np.int64)]
    vals = flat[:, idx].astype(np.float32)
    top = float(np.percentile(vals, 99.995))
    top = top if top > 0 else 1.0
    stats = [_robust(vals[i] / top) for i in range(c)]
    if linked:
        med = float(np.mean([s[0] for s in stats]))
        mad = float(np.mean([s[1] for s in stats]))
        stats = [(med, mad)] * c
    out = []
    for med, mad in stats:
        c0 = min(max(med + clip * mad, 0.0), 1.0)
        x = (med - c0) / (1.0 - c0) if c0 < 1 else 0.5
        m = float(mtf(target, x)) if 0 < x < 1 else 0.5
        out.append((c0 * top, m, top))
    return out


def apply_stf(data: np.ndarray, params) -> np.ndarray:
    """(C, h, w) linear -> (h, w, 3) uint8 RGB with the given per-channel (shadows, midtones, highlights)."""
    c, h, w = data.shape
    rgb = np.empty((h, w, 3), np.uint8)
    chans = [0, 0, 0] if c == 1 else [0, 1, 2]
    for o, i in enumerate(chans):
        lo, m, hi = params[i]
        x = np.clip((data[i] - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        rgb[..., o] = np.clip(mtf(m, x) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return rgb


COVERAGE_STOPS = [(0.0, (14, 18, 32)), (0.22, (38, 64, 140)), (0.45, (40, 150, 168)),
                  (0.70, (226, 196, 84)), (1.0, (255, 248, 230))]


def coverage_lut() -> np.ndarray:
    x = np.linspace(0, 1, 256)
    pos = [p for p, _ in COVERAGE_STOPS]
    return np.stack([np.interp(x, pos, [c[k] for _, c in COVERAGE_STOPS]) for k in range(3)], axis=1).astype(np.uint8)


def coverage_rgba(depth: np.ndarray, vmax: int) -> np.ndarray:
    """Depth map -> (h, w, 4) uint8, sqrt-scaled colors, transparent where no frame covers."""
    lut = coverage_lut()
    t = np.sqrt(np.clip(depth / max(vmax, 1), 0, 1))
    idx = (t * 255).astype(np.int32)
    rgba = np.empty(depth.shape + (4,), np.uint8)
    rgba[..., :3] = lut[idx]
    rgba[..., 3] = np.where(depth > 0, 255, 0)
    return rgba
