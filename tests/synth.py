# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Synthetic Siril-style registered sequences for tests: r_ frames, drizztmp weights and a .seq."""
from __future__ import annotations

import os

import numpy as np

from gpustacker.engine import fits as F


def star_field(h: int, w: int, rng, n: int = 40) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        a = rng.uniform(0.02, 0.4)
        img += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6**2))
    return img


def make_sequence(directory: str, n: int = 12, w: int = 96, h: int = 72, layers: int = 3, canvas=(160, 120),
                  drizzle: bool = True, blank: tuple = (), excluded: tuple = (), seed: int = 0,
                  gains=None, offsets=None, exptimes=None, outliers: bool = True, corner_cut: int = 10,
                  name: str = "r_pp_lights_") -> str:
    """Write n registered frames cut from one noisy canvas scene; returns the .seq path.

    Frame i = (scene[canvas window] * gain_i + offset_i + noise), zero outside its rotated-looking
    corners; registration translations place it back on the canvas (top-down, like Siril)."""
    rng = np.random.default_rng(seed)
    cw, ch = canvas
    scene = np.stack([0.05 + star_field(ch, cw, rng) * (1.0 + 0.1 * c) for c in range(layers)])
    os.makedirs(os.path.join(directory, "drizztmp"), exist_ok=True)
    gains = gains if gains is not None else rng.uniform(0.8, 1.25, n)
    offsets = offsets if offsets is not None else rng.uniform(-0.01, 0.02, n)
    exptimes = exptimes if exptimes is not None else [60.0 if i % 3 else 30.0 for i in range(n)]
    lines_i, lines_r = [], []
    for i in range(n):
        x0, y0 = int(rng.integers(0, cw - w + 1)), int(rng.integers(0, ch - h + 1))
        data = scene[:, y0:y0 + h, x0:x0 + w] * gains[i] + offsets[i]
        data = data + rng.normal(0, 0.004 * np.sqrt(60.0 / exptimes[i]), data.shape).astype(np.float32)
        if outliers and i % 4 == 1:                      # a satellite trail and hot pixels
            r = int(rng.integers(5, h - 5))
            data[:, r:r + 2, :] += 0.5
            data[:, rng.integers(0, h, 6), rng.integers(0, w, 6)] = 0.95
        valid = np.ones((h, w), bool)
        k = corner_cut
        for yy in range(k):                              # zeroed corners, like rotated frames
            valid[yy, :k - yy] = False
            valid[h - 1 - yy, w - k + yy:] = False
        if i in blank:
            valid[:] = False
            valid[h // 2, w // 2] = True
        data = np.where(valid[None], np.clip(data, 1e-4, 1.0), 0.0).astype(np.float32)
        filenum = i + 1
        hdr = [F.card("EXPTIME", float(exptimes[i])), F.card("DATE-OBS", f"2026-10-0{1 + i // 10}T22:{i % 60:02d}:00"),
               F.card("FILTER", "LP"), F.card("OBJECT", "Synthetic")]
        if i == 0:
            hdr += [F.card("CTYPE1", "RA---TAN"), F.card("CTYPE2", "DEC--TAN"), F.card("CRPIX1", 48.5),
                    F.card("CRPIX2", 36.5), F.card("CRVAL1", 10.68), F.card("CRVAL2", 41.27),
                    F.card("CD1_1", -0.000624), F.card("CD2_2", 0.000624)]
        F.write_fits(os.path.join(directory, f"{name}{filenum:05d}.fit"), data, hdr)
        if drizzle:
            dz = np.where(valid, 1, 0).astype(np.uint8)
            _write_u8(os.path.join(directory, "drizztmp", f"{name}{filenum:05d}.fit"),
                      np.repeat(dz[None], layers, axis=0))
        lines_i.append(f"I {filenum} {0 if i in excluded else 1} {w},{h}")
        # Siril: canvas row of the frame top = H[1][2] - min(H[1][2]); translation_from_H gives dy = -H[1][2]
        lines_r.append(f"R1 3.2 3.2 0.8 0 0.05 {int(rng.integers(50, 300))} H 1 0 {x0} 0 1 {y0} 0 0 1")
    seq_path = os.path.join(directory, f"{name}.seq")
    nsel = n - len(excluded)
    with open(seq_path, "w") as f:
        f.write("#Siril sequence file. Contains list of images, selection, registration data and statistics\n")
        f.write("#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version "
                "variable_size fz_flag drizzle\n")
        f.write(f"S '{name}' 1 {n} {nsel} 5 0 6 1 0 {1 if drizzle else 0}\nL {layers}\n")
        f.write("\n".join(lines_i) + "\n" + "\n".join(lines_r) + "\n")
    return seq_path


def _write_u8(path: str, data: np.ndarray) -> None:
    c, h, w = data.shape
    head = [F.card("SIMPLE", True), F.card("BITPIX", 8), F.card("NAXIS", 3), F.card("NAXIS1", w),
            F.card("NAXIS2", h), F.card("NAXIS3", c), "END".ljust(80)]
    text = "".join(head)
    text = text.ljust(-(-len(text) // F.BLOCK) * F.BLOCK)
    body = np.ascontiguousarray(data[:, ::-1, :]).tobytes()
    with open(path, "wb") as f:
        f.write(text.encode("ascii"))
        f.write(body)
        f.write(b"\0" * (-len(body) % F.BLOCK))
