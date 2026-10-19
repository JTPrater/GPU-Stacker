# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Normalization statistics vs straight NumPy ports of Siril 1.4.4's C code (needs an NVIDIA GPU + CuPy).

The ports below follow Siril line by line, in float32 where it computes in float; the GPU code must give the
very same numbers. (The comparison with Siril itself is tools\\check_siril_script.py.)
"""
import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpustacker.engine.stats import histogram_median, mad, measure_planes, norm_stats, sequential_sums  # noqa: E402


def _hist_median_cpu(a):
    """RawTherapee's findMinMaxPercentile(data, n, 0.5, ...) = Siril's histogram_median_float."""
    a = np.asarray(a, np.float32)
    n = a.size
    lo, hi = a.min(), a.max()
    if abs(hi - lo) == 0:
        return float(lo)
    nbins = min(65536, n)
    scale = np.float32(nbins - 1) / (hi - lo)
    with np.errstate(invalid="ignore"):
        b = ((a - lo) * scale).astype(np.int32).astype(np.uint16)        # static_cast<uint16_t>
    h = np.bincount(b, minlength=nbins)
    cum = np.cumsum(h)
    thresh = np.float32(0.5) * np.float32(n)
    j = int(np.argmax(cum.astype(np.float32) >= thresh))                  # while (count < thresh) ...
    k, count, below = j + 1, cum[j], cum[j] - h[j]
    c0 = np.float32(count) - thresh
    c1 = thresh - np.float32(below)
    m = (c1 * np.float32(k) + c0 * np.float32(k - 1)) / (c0 + c1)
    m = m / scale
    m = m + lo
    return float(max(lo, min(m, hi)))


def _norm_stats_cpu(x):
    """statistics_internal_float(STATS_NORM): histogram median and MAD, then IKSSlite with float bwmv terms
    added one by one in double."""
    x = np.asarray(x, np.float32)
    m = _hist_median_cpu(x)
    s = _hist_median_cpu(np.abs(x - np.float32(m)))
    k = x[(x >= np.float32(m - 6.0 * s)) & (x <= np.float32(m + 6.0 * s))]
    loc = _hist_median_cpu(k)
    s = _hist_median_cpu(np.abs(k - np.float32(loc)))
    factor = np.float32(1) / (np.float32(9) * np.float32(s))
    i_med = k - np.float32(loc)
    yi = i_med * factor
    yi2 = np.where(np.abs(yi) < 1, yi * yi, np.float32(1))
    w = np.float32(1) - yi2
    a = i_med * (w * w)
    up = np.cumsum(a * a, dtype=np.float64)[-1]
    down = np.cumsum(w * (np.float32(1) - np.float32(5) * yi2), dtype=np.float64)[-1]
    return loc, math.sqrt(k.size * (up / (down * down))) * .991


def _frame(rng, h=300, w=420):
    """Seestar-like plane: background with gradient and noise, stars, hot pixels, an empty (zero) border."""
    yy, xx = np.mgrid[:h, :w]
    img = 0.06 + 0.00002 * xx + rng.normal(0, 0.004, (h, w))
    for _ in range(40):
        cy, cx, f = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0.05, 0.9)
        img += f * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.5)
    img[rng.integers(0, h, 30), rng.integers(0, w, 30)] = 1.0
    img[:, :17] = 0
    img[-9:, :] = 0
    return img.astype(np.float32)


def test_histogram_median_matches_rawtherapee_port():
    rng = np.random.default_rng(11)
    cases = [rng.normal(0.05, 0.004, 300_000), np.concatenate([rng.normal(0.05, 0.004, 90_000), rng.random(300)]),
             [0.1, 0.3], [0.3, 0.1, 0.2], [0.7], np.full(500, 0.25), rng.random(10), rng.random(65535),
             rng.random(65536), rng.random(65537), rng.normal(0, 1, 100_000), rng.integers(0, 50, 200_000),
             rng.exponential(1.0, 200_000), 1.0 + 1e-6 * rng.random(50_000)]
    for a in cases:
        a = np.asarray(a, np.float32)
        assert histogram_median(cp.asarray(a)) == _hist_median_cpu(a)
        assert mad(cp.asarray(a), 0.0123) == _hist_median_cpu(np.abs(a - np.float32(0.0123)))


def test_sequential_sums_match_a_c_loop():
    rng = np.random.default_rng(12)
    d = rng.normal(0, 0.004, 1_500_000).astype(np.float32)
    rows = [
        (d * d) ** 2,                                                 # bwmv-like: thousands of ties
        1 - 5 * (d / 0.03) ** 2,                                      # mixed signs
        np.concatenate([rng.random(5000), [2.0 ** 40], rng.integers(-6, 7, 600_000) * 2.0 ** -13]),   # all ties
        np.where(rng.random(400_000) < 0.5, 0, rng.random(400_000)),
        rng.normal(0, 1, 300_000) * 10.0 ** rng.integers(-20, 5, 300_000),  # wide magnitudes, sign changes
        rng.random(4097), rng.random(100), [],
        np.where(rng.random(20_000) < 1e-3, np.nan, rng.random(20_000)),   # non-finite: CPU fallback
    ]
    for t in rows:
        t = np.asarray(t, np.float32)[None]
        want = np.cumsum(t, axis=1, dtype=np.float64)[:, -1] if t.size else [0.0]
        got = sequential_sums(cp.asarray(t))
        assert np.array_equal(np.asarray(got), np.asarray(want), equal_nan=True)
    two = rng.random((2, 30_000)).astype(np.float32)                 # several rows at once
    assert sequential_sums(cp.asarray(two)) == np.cumsum(two, axis=1, dtype=np.float64)[:, -1].tolist()


def test_norm_stats_match_siril_port():
    rng = np.random.default_rng(13)
    planes = np.stack([_frame(rng) for _ in range(3)])
    planes[2] *= 0.8                                                 # another level per channel
    st = measure_planes(cp.asarray(planes), need_norm=True, need_noise=False)
    for c in range(3):
        x = planes[c][planes[c] != 0]
        assert st.ngood[c] == x.size
        assert (st.loc[c], st.scale[c]) == _norm_stats_cpu(x)        # bit-exact, not just close
        assert norm_stats(cp.asarray(x)) == _norm_stats_cpu(x)
