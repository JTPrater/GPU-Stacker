# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
# Contains code ported from Siril and OpenCV; see THIRD-PARTY-NOTICES.txt
"""Feather masks: Siril's cached .msk files, or the same masks computed here.

Siril (stacking/blending.c + opencv.cpp cvDownscaleBlendMask), per frame:
  1. 255 where the green layer (mono: layer 0) is non-zero, else 0, full resolution
  2. closing with a 7x7 rectangle (dilate, then erode) to fill drizzle holes
  3. INTER_LINEAR resize to (int)(w * 0.1) x (int)(h * 0.1), inside a 1-pixel black border
  4. distanceTransform(DIST_L2, 3x3 mask, CV_32F) and the border cropped off
The file keeps Siril's in-memory row order, i.e. FITS order (bottom row first).
"""
from __future__ import annotations

import os

import numpy as np

from .fits import read_header, write_fits

MASK_SCALE = 0.1
_SHIFT = 16
_HV = int(np.rint(np.float32(0.955) * np.float32(1 << _SHIFT)))      # cvRound(0.955f * 65536) = 62587
_DIAG = int(np.rint(np.float32(1.3693) * np.float32(1 << _SHIFT)))   # 89738
_INIT = 2**31 - 1
_DIST_MAX = _INIT >> 2


def mask_size(width: int, height: int) -> tuple[int, int]:
    """Siril's compute_downscaled_mask_size: (int)(rx * 0.1), (int)(ry * 0.1)."""
    return int(width * MASK_SCALE), int(height * MASK_SCALE)


def read_mask(path: str) -> np.ndarray:
    """A .msk file as top-down float32 (mh, mw)."""
    hdr = read_header(path)
    _, mh, mw = hdr.shape
    raw = np.fromfile(path, dtype=hdr.dtype, count=mh * mw, offset=hdr.data_offset).reshape(mh, mw)
    return np.ascontiguousarray(raw[::-1], dtype=np.float32)


def write_mask(path: str, mask_topdown: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_fits(path, mask_topdown[None].astype(np.float32), [])


def chamfer_distance(fg: np.ndarray) -> np.ndarray:
    """OpenCV distanceTransform(DIST_L2, 3, CV_32F): distance of each non-zero pixel to the nearest
    zero pixel with the 3x3 chamfer metric (0.955, 1.3693), 16-bit fixed point, two passes."""
    h, w = fg.shape
    fgm = fg != 0
    tmp = np.full((h + 2, w + 2), _INIT, np.int64)          # one-pixel border of INIT_DIST0
    j = np.arange(w, dtype=np.int64)
    left = (_INIT + _HV) + j * _HV                          # path entering from the left border
    for i in range(h):                                      # forward pass
        up = tmp[i]
        c = np.minimum(np.minimum(up[0:w] + _DIAG, up[1:w + 1] + _HV), up[2:w + 2] + _DIAG)
        c[~fgm[i]] = 0
        row = np.minimum(np.minimum.accumulate(c - j * _HV) + j * _HV, left)
        row[~fgm[i]] = 0
        tmp[i + 1, 1:w + 1] = row
    out = np.empty((h, w), np.float32)
    right = (_INIT + w * _HV) - j * _HV                      # path entering from the right border
    for i in range(h - 1, -1, -1):                          # backward pass
        down = tmp[i + 2]
        e = np.minimum(tmp[i + 1, 1:w + 1],
                       np.minimum(np.minimum(down[2:w + 2] + _DIAG, down[1:w + 1] + _HV), down[0:w] + _DIAG))
        hk = e + j * _HV
        row = np.minimum(np.minimum.accumulate(hk[::-1])[::-1] - j * _HV, right)
        tmp[i + 1, 1:w + 1] = row
        out[i] = np.minimum(row, _DIST_MAX).astype(np.float32) * np.float32(1.0 / (1 << _SHIFT))
    return out


def _linear_coeffs(src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV resize INTER_LINEAR source offsets and 11-bit fixed-point weights."""
    scale = 1.0 / (dst / src)
    f = ((np.arange(dst) + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.int64)
    f = (f - s.astype(np.float32)).astype(np.float32)
    lo = s < 0
    f[lo], s[lo] = 0, 0
    hi = s >= src - 1
    f[hi], s[hi] = 0, src - 1
    c0 = np.rint((np.float32(1) - f) * np.float32(2048))
    c1 = np.rint(f * np.float32(2048))
    coef = np.stack([c0, c1], axis=1).astype(np.int16).ravel()
    return s.astype(np.int32), coef


def make_mask(plane):
    """Siril's downscaled distance mask from one layer (CuPy (h, w) array in FITS row order).

    Returns float32 (mh, mw) in the same row order as the input."""
    return finish_mask(downscaled_mask(plane))


def finish_mask(padded: np.ndarray) -> np.ndarray:
    """CPU half of make_mask (thread-safe): distance transform of the padded downscaled mask."""
    return chamfer_distance(padded)[1:-1, 1:-1].copy()


def downscaled_mask(plane) -> np.ndarray:
    """GPU half of make_mask: closing, INTER_LINEAR downscale, 1-pixel black border (host uint8)."""
    import cupy as cp
    import cupyx.scipy.ndimage as ndi

    from .kernels import launch, resize_u8_linear

    h, w = plane.shape
    b = cp.where(plane != 0, cp.uint8(255), cp.uint8(0))
    b = ndi.maximum_filter(b, size=7, mode="nearest")
    b = ndi.minimum_filter(b, size=7, mode="nearest")
    mw, mh = mask_size(w, h)
    if mw < 1 or mh < 1:
        raise ValueError(f"frame too small for a feather mask ({w} x {h})")
    xofs, ax = _linear_coeffs(w, mw)
    yofs, by = _linear_coeffs(h, mh)
    down = cp.empty((mh, mw), cp.uint8)
    launch(resize_u8_linear, mw * mh, cp.ascontiguousarray(b), np.int32(w), np.int32(h), np.int32(mw),
           np.int32(mh), cp.asarray(xofs), cp.asarray(ax), cp.asarray(yofs), cp.asarray(by), down)
    padded = np.zeros((mh + 2, mw + 2), np.uint8)
    padded[1:-1, 1:-1] = down.get()
    return padded
