# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Star detection on the GPU, for registration and per-frame quality metrics.

Works on a 2x2-binned luminance (each 2x2 CFA block holds one R, two G and one B, so binning
is colour-neutral and hot pixels shrink to nothing): mesh background, MAD noise, 5-sigma local
maxima of a lightly smoothed residual, then moments in a small window for centroid, flux,
FWHM and roundness. Positions are returned in full-resolution top-down continuous coordinates
(pixel x covers [x, x+1)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MESH = 32            # background cell, binned pixels
MAX_CANDIDATES = 4000
KEEP = 800           # brightest stars kept per frame


@dataclass
class StarList:
    x: np.ndarray          # full-res top-down continuous coordinates
    y: np.ndarray
    flux: np.ndarray
    fwhm: np.ndarray       # full-res pixels
    roundness: np.ndarray  # minor / major axis (1 = round)
    saturated: np.ndarray  # bool

    def __len__(self) -> int:
        return len(self.x)

    def brightest(self, n: int) -> "StarList":
        o = np.argsort(-self.flux)[:n]
        return StarList(self.x[o], self.y[o], self.flux[o], self.fwhm[o], self.roundness[o], self.saturated[o])

    def to_array(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.flux, self.fwhm, self.roundness, self.saturated.astype(np.float32)], 1)

    @staticmethod
    def from_array(a: np.ndarray) -> "StarList":
        a = np.asarray(a, dtype=np.float64).reshape(-1, 6)
        return StarList(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5] > 0.5)


@dataclass
class FrameMetrics:
    nstars: int
    fwhm: float            # median, full-res pixels
    roundness: float
    background: float      # median background level (0..1)
    noise: float           # background sigma of the binned luminance


_MEASURE = r"""
extern "C" __global__ void measure(const float* __restrict__ R, const float* __restrict__ L, const int H, const int W,
                                   const int* __restrict__ cx, const int* __restrict__ cy, const int n,
                                   const float thresh, const float satlevel, float* __restrict__ out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int x0 = cx[k], y0 = cy[k];
    const int rad = 4;
    double s = 0, sx = 0, sy = 0;
    float peak = 0.f, lmax = 0.f;
    for (int dy = -rad; dy <= rad; dy++) {
        const int y = y0 + dy;
        if (y < 0 || y >= H) continue;
        for (int dx = -rad; dx <= rad; dx++) {
            const int x = x0 + dx;
            if (x < 0 || x >= W || dx * dx + dy * dy > rad * rad + 1) continue;
            const float v = R[(long long)y * W + x];
            lmax = fmaxf(lmax, L[(long long)y * W + x]);
            if (v > peak) peak = v;
            if (v > thresh) { s += v; sx += v * (x + 0.5); sy += v * (y + 0.5); }
        }
    }
    float* o = out + 8 * (long long)k;
    if (s <= 0) { o[0] = -1.f; return; }
    const double mx = sx / s, my = sy / s;
    double mxx = 0, myy = 0, mxy = 0, w = 0;
    for (int dy = -rad; dy <= rad; dy++) {
        const int y = y0 + dy;
        if (y < 0 || y >= H) continue;
        for (int dx = -rad; dx <= rad; dx++) {
            const int x = x0 + dx;
            if (x < 0 || x >= W || dx * dx + dy * dy > rad * rad + 1) continue;
            const float v = R[(long long)y * W + x];
            if (v > thresh) {
                const double ddx = x + 0.5 - mx, ddy = y + 0.5 - my;
                mxx += v * ddx * ddx; myy += v * ddy * ddy; mxy += v * ddx * ddy; w += v;
            }
        }
    }
    mxx /= w; myy /= w; mxy /= w;
    const double tr = mxx + myy, det = mxx * myy - mxy * mxy;
    const double disc = sqrt(fmax(tr * tr / 4 - det, 0.0));
    const double l1 = tr / 2 + disc, l2 = fmax(tr / 2 - disc, 1e-9);
    o[0] = (float)mx; o[1] = (float)my; o[2] = (float)s; o[3] = (float)(2.3548 * sqrt((l1 + l2) / 2));
    o[4] = (float)sqrt(l2 / l1); o[5] = peak; o[6] = lmax >= satlevel ? 1.f : 0.f; o[7] = (float)w;
}
"""
_mod = None


def _kernel():
    global _mod
    if _mod is None:
        import cupy as cp
        _mod = cp.RawModule(code=_MEASURE).get_function("measure")
    return _mod


def bin2(img):
    """(H, W) CuPy -> (H//2, W//2) mean of 2x2 blocks (top-down, even-aligned)."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def luminance(planes, cfa: bool):
    """Binned luminance from a decoded frame (planes, H, W)."""
    if cfa:
        return bin2(planes[0])
    return bin2(planes.mean(axis=0))


def _background(L):
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
    h, w = L.shape
    ny, nx = max(1, h // MESH), max(1, w // MESH)
    cells = L[:ny * MESH, :nx * MESH].reshape(ny, MESH, nx, MESH).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    mesh = cp.median(cells, axis=-1)
    if ny >= 3 and nx >= 3:
        mesh = ndi.median_filter(mesh, size=3, mode="nearest")
    zy = cp.linspace(0, ny - 1, h, dtype=cp.float32)
    zx = cp.linspace(0, nx - 1, w, dtype=cp.float32)
    yy, xx = cp.meshgrid(zy, zx, indexing="ij")
    return ndi.map_coordinates(mesh.astype(cp.float32), cp.stack([yy, xx]), order=1, mode="nearest")


def detect(L, saturation: float = 0.98) -> tuple[StarList, FrameMetrics]:
    """Stars in a binned luminance image L (CuPy float32, values 0..1)."""
    import cupy as cp
    import cupyx.scipy.ndimage as ndi

    h, w = L.shape
    bg = _background(L)
    R = (L - bg).astype(cp.float32)
    samp = R[::3, ::3].ravel()
    med = float(cp.median(samp))
    sigma = float(cp.median(cp.abs(samp - med))) * 1.4826 or 1e-6
    S = ndi.gaussian_filter(R, 1.0)
    ss = S[::3, ::3].ravel()
    s_med = float(cp.median(ss))
    s_sig = float(cp.median(cp.abs(ss - s_med))) * 1.4826 or 1e-6
    peaks = (S > s_med + 5.0 * s_sig) & (S == ndi.maximum_filter(S, size=5))
    m = 6
    peaks[:m, :] = False
    peaks[-m:, :] = False
    peaks[:, :m] = False
    peaks[:, -m:] = False
    ys, xs = cp.nonzero(peaks)
    if ys.size == 0:
        return _empty(), FrameMetrics(0, float("nan"), float("nan"), float(cp.median(bg)), sigma)
    if ys.size > MAX_CANDIDATES:
        top = cp.argsort(-S[ys, xs])[:MAX_CANDIDATES]
        ys, xs = ys[top], xs[top]
    n = int(ys.size)
    out = cp.empty((n, 8), cp.float32)
    _kernel()(((n + 127) // 128,), (128,), (R, L, np.int32(h), np.int32(w), xs.astype(cp.int32),
                                          ys.astype(cp.int32), np.int32(n), np.float32(2.0 * sigma),
                                          np.float32(saturation), out))
    o = out.get()
    o = o[o[:, 0] >= 0]
    # half-res (binned) continuous coords -> full resolution
    fwhm = o[:, 3] * 2.0
    ok = (o[:, 3] > 0.55) & (o[:, 4] > 0.25) & np.isfinite(fwhm)       # drop hot pixels / streaks
    o, fwhm = o[ok], fwhm[ok]
    if len(o) == 0:
        return _empty(), FrameMetrics(0, float("nan"), float("nan"), float(cp.median(bg)), sigma)
    med_fwhm = float(np.median(fwhm))
    ok = fwhm < 3.0 * med_fwhm + 2.0                                     # drop galaxies / blobs
    o, fwhm = o[ok], fwhm[ok]
    total = len(o)
    order = np.argsort(-o[:, 2])[:KEEP]
    o, fwhm = o[order], fwhm[order]
    stars = StarList(x=o[:, 0].astype(np.float64) * 2.0, y=o[:, 1].astype(np.float64) * 2.0, flux=o[:, 2].astype(np.float64),
                     fwhm=fwhm.astype(np.float64), roundness=o[:, 4].astype(np.float64), saturated=o[:, 6] > 0.5)
    good = ~stars.saturated & (stars.roundness > 0.4)
    use = good if good.sum() >= 5 else np.ones(len(stars), bool)
    metrics = FrameMetrics(nstars=int(total), fwhm=float(np.median(stars.fwhm[use])),
                           roundness=float(np.median(stars.roundness[use])),
                           background=float(cp.median(bg)), noise=sigma)
    return stars, metrics


def _empty() -> StarList:
    z = np.zeros(0)
    return StarList(z, z, z, z, z, np.zeros(0, bool))
