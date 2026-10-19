# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Synthetic raw lights for tests: one star field seen through known transforms, written like the
Seestar's subs (16-bit CFA with BZERO 32768 and BAYERPAT, rows stored top-down, no ROWORDER).

Coordinates follow the engine: top-down continuous pixels, pixel x covers [x, x+1). Each frame's
transform maps its own coordinates to frame 0's (the reference)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from gpustacker.engine import fits as F
from gpustacker.engine.lights import cfa_lut

COLOR = (1.0, 0.7, 0.35)          # star brightness in R, G, B (per unit of `peak`)


@dataclass
class Truth:
    lights_dir: str
    names: list                   # file names, in time order
    transforms: list              # per frame: 2x3 frame -> frame 0
    stars: np.ndarray             # (n, 3): x, y, peak in frame-0 coordinates
    width: int
    height: int
    sky: float
    noise: float
    exptime: float
    trail: tuple | None = None    # (frame index, (x0, y0), (x1, y1)) in that frame's coordinates


def similarity(rot_deg: float, dx: float, dy: float, cx: float, cy: float) -> np.ndarray:
    """Frame -> reference: rotate by rot_deg about (cx, cy), then shift by (dx, dy)."""
    th = np.radians(rot_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, cx + dx - (c * cx - s * cy)], [s, c, cy + dy - (s * cx + c * cy)]])


def apply(m: np.ndarray, x, y):
    return m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]


def invert(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.vstack([m, [0.0, 0.0, 1.0]]))[:2]


def make_lights(directory: str, n: int = 10, w: int = 320, h: int = 240, pattern: str = "GRBG", seed: int = 0,
                density: float = 220 / (320 * 240), fwhm: float = 3.2, sky: float = 0.06, noise: float = 0.004,
                offsets=None, rotations=None, cloudy=(), trail: int | None = None, exptime: float = 10.0,
                bottom_up: bool = False) -> Truth:
    """Write n lights into <directory>/lights and return what they contain."""
    rng = np.random.default_rng(seed)
    lights_dir = os.path.join(directory, "lights")
    os.makedirs(lights_dir, exist_ok=True)
    if offsets is None:
        offsets = [(0.0, 0.0)] + [tuple(rng.uniform(-12, 12, 2)) for _ in range(n - 1)]
    if rotations is None:
        rotations = [0.0] + list(rng.uniform(-0.6, 0.6, n - 1))
    ts = [similarity(rotations[i], offsets[i][0], offsets[i][1], w / 2, h / 2) for i in range(n)]

    # stars over everything any frame sees (frame-0 coordinates), log-uniform peaks 0.01 .. 0.5
    cx, cy = [], []
    for m in ts:
        x, y = apply(m, np.array([0.0, w, w, 0.0]), np.array([0.0, 0.0, h, h]))
        cx += list(x)
        cy += list(y)
    x0, x1, y0, y1 = min(cx) - 20, max(cx) + 20, min(cy) - 20, max(cy) + 20
    count = int(density * (x1 - x0) * (y1 - y0))
    stars = np.stack([rng.uniform(x0, x1, count), rng.uniform(y0, y1, count),
                      0.01 * 50.0 ** rng.uniform(0, 1, count)], 1)

    lut = cfa_lut(pattern)
    yy, xx = np.mgrid[0:h, 0:w]
    chan = lut[yy % 2, xx % 2]
    sig = fwhm / 2.3548
    r = int(np.ceil(4 * sig))
    names = [f"Light_Synthetic_{exptime:.1f}s_IRCUT_{i:04d}.fit" for i in range(n)]
    trail_info = None
    for i, m in enumerate(ts):
        inv = invert(m)
        fx, fy = apply(inv, stars[:, 0], stars[:, 1])
        peak = stars[:, 2] * (0.01 if i in cloudy else 1.0)
        lum = np.zeros((h, w), np.float64)
        for x, y, p in zip(fx, fy, peak):
            if not (-r <= x < w + r and -r <= y < h + r):
                continue
            ia, ib = max(0, int(y) - r), min(h, int(y) + r + 1)
            ja, jb = max(0, int(x) - r), min(w, int(x) + r + 1)
            if ia >= ib or ja >= jb:
                continue
            gy = (np.arange(ia, ib) + 0.5 - y)[:, None]
            gx = (np.arange(ja, jb) + 0.5 - x)[None, :]
            lum[ia:ib, ja:jb] += p * np.exp(-(gx * gx + gy * gy) / (2 * sig * sig))
        bg = sky * (3.0 if i in cloudy else 1.0)
        img = bg + np.choose(chan, [COLOR[0] * lum, COLOR[1] * lum, COLOR[2] * lum])
        if trail is not None and i == trail:                 # a 2 px wide satellite trail, corner to corner
            a, b = (0.0, h * 0.2), (float(w), h * 0.75)
            d = np.abs((b[1] - a[1]) * (xx + 0.5 - a[0]) - (b[0] - a[0]) * (yy + 0.5 - a[1])) / np.hypot(
                b[0] - a[0], b[1] - a[1])
            img = img + np.where(d <= 1.0, 0.25, 0.0)
            trail_info = (i, a, b)
        img = img + rng.normal(0, noise, img.shape)
        u16 = np.clip(np.round(img * 65535.0), 0, 65535).astype(np.uint16)
        date = f"2026-10-11T03:{i // 5:02d}:{(i % 5) * 12:02d}.000"
        _write_raw(os.path.join(lights_dir, names[i]), u16, pattern, exptime, date, bottom_up)
    return Truth(lights_dir, names, ts, stars, w, h, sky, noise, exptime, trail_info)


def _write_raw(path: str, u16: np.ndarray, pattern: str, exptime: float, date: str, bottom_up: bool) -> None:
    h, w = u16.shape
    cards = [F.card("SIMPLE", True), F.card("BITPIX", 16), F.card("NAXIS", 2), F.card("NAXIS1", w),
             F.card("NAXIS2", h), F.card("BZERO", 32768), F.card("BSCALE", 1),
             F.card("EXPTIME", float(exptime)), F.card("DATE-OBS", date), F.card("FILTER", "IRCUT"),
             F.card("OBJECT", "Synthetic"), F.card("BAYERPAT", pattern), F.card("XPIXSZ", 2.9),
             F.card("INSTRUME", "Synthetic S50")]
    rows = u16
    if bottom_up:
        cards.append(F.card("ROWORDER", "BOTTOM-UP"))
        rows = u16[::-1]
    text = "".join(cards) + "END".ljust(80)
    text = text.ljust(-(-len(text) // F.BLOCK) * F.BLOCK)
    body = (rows.astype(np.int32) - 32768).astype(">i2").tobytes()
    with open(path, "wb") as f:
        f.write(text.encode("ascii"))
        f.write(body)
        f.write(b"\0" * (-len(body) % F.BLOCK))
