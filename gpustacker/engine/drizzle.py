# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Drizzle on the GPU (square kernel), including Bayer drizzle straight from the CFA mosaic.

Like Siril's drizzle (a port of STScI drizzle): each input pixel's square footprint, shrunk by
pixfrac around its centre, is mapped through the registration transform, and its exact overlap
area with every output pixel ("boxer") is accumulated as value * area and area. The output
pixel is the weighted mean (num / den); den is the drizzle weight, which Siril stores as
round(den * 255) in drizztmp. With a CFA, each raw pixel only feeds its own colour's plane.
Input and output coordinates are top-down continuous pixels (pixel x covers [x, x+1)).
"""
from __future__ import annotations

import math

import numpy as np

_SRC = r"""
// signed area between segment (x1,y1)-(x2,y2) and the x axis, clipped to the unit square
__device__ float sgarea(float x1, float y1, float x2, float y2) {
    const float dx = x2 - x1;
    if (dx == 0.0f) return 0.0f;
    const bool negdx = dx < 0.0f;
    float xlo = negdx ? x2 : x1, xhi = negdx ? x1 : x2;
    if (xlo >= 1.0f || xhi <= 0.0f) return 0.0f;
    xlo = fmaxf(xlo, 0.0f);
    xhi = fminf(xhi, 1.0f);
    const float m = (y2 - y1) / dx;
    const float c = y1 - m * x1;
    float ylo = m * xlo + c, yhi = m * xhi + c;
    if (ylo <= 0.0f && yhi <= 0.0f) return 0.0f;
    if (ylo < 0.0f) { ylo = 0.0f; xlo = -c / m; }
    if (yhi < 0.0f) { yhi = 0.0f; xhi = -c / m; }
    float a;
    if (ylo >= 1.0f && yhi >= 1.0f) {
        a = xhi - xlo;
    } else if (ylo <= 1.0f && yhi <= 1.0f) {
        a = 0.5f * (xhi - xlo) * (yhi + ylo);
    } else {
        const float xtop = (1.0f - c) / m;
        if (ylo <= 1.0f) a = 0.5f * (xtop - xlo) * (1.0f + ylo) + xhi - xtop;
        else a = 0.5f * (xhi - xtop) * (1.0f + yhi) + xtop - xlo;
    }
    return negdx ? -a : a;
}

// One thread per input pixel. img: (nrows, W) rows y0.. of the frame (top-down), values 0..1.
// lut: CFA channel of (y&1, x&1), or nullptr-like flag cfa=0 for single-plane input.
// only_ch >= 0: skip pixels of other colours and write plane 0; else write plane = colour.
extern "C" __global__ void drizzle(const float* __restrict__ img, const int W, const int nrows, const int y0,
                                   const int cfa, const int l00, const int l01, const int l10, const int l11,
                                   const int only_ch,
                                   const double a, const double b, const double c,
                                   const double d, const double e, const double f,
                                   const double half, const int ow, const int oh,
                                   float* __restrict__ num, float* __restrict__ den) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)W * nrows) return;
    const int x = (int)(i % W), y = y0 + (int)(i / W);
    int ch = 0;
    if (cfa) {
        const int k = ((y & 1) << 1) | (x & 1);
        ch = k == 0 ? l00 : (k == 1 ? l01 : (k == 2 ? l10 : l11));
        if (only_ch >= 0 && ch != only_ch) return;
    }
    const int plane = only_ch >= 0 ? 0 : ch;
    const float v = img[i];
    const double cx = x + 0.5, cy = y + 0.5;
    const double px[4] = {cx - half, cx + half, cx + half, cx - half};
    const double py[4] = {cy - half, cy - half, cy + half, cy + half};
    double qx[4], qy[4];
    double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
    for (int k = 0; k < 4; k++) {
        qx[k] = a * px[k] + b * py[k] + c;
        qy[k] = d * px[k] + e * py[k] + f;
        minx = fmin(minx, qx[k]); maxx = fmax(maxx, qx[k]);
        miny = fmin(miny, qy[k]); maxy = fmax(maxy, qy[k]);
    }
    const int u0 = max((int)floor(minx), 0), u1 = min((int)ceil(maxx) - 1, ow - 1);
    const int v0 = max((int)floor(miny), 0), v1 = min((int)ceil(maxy) - 1, oh - 1);
    if (u0 > u1 || v0 > v1) return;
    float* nump = num + (long long)plane * ow * oh;
    float* denp = den + (long long)plane * ow * oh;
    for (int vv = v0; vv <= v1; vv++) {
        for (int uu = u0; uu <= u1; uu++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) {
                const int k2 = (k + 1) & 3;
                s += sgarea((float)(qx[k] - uu), (float)(qy[k] - vv), (float)(qx[k2] - uu), (float)(qy[k2] - vv));
            }
            const float area = fabsf(s);
            if (area > 0.0f) {
                const long long o = (long long)vv * ow + uu;
                atomicAdd(nump + o, area * v);
                atomicAdd(denp + o, area);
            }
        }
    }
}

// num/den -> value, uint8 drizzle weight round(min(den,1) * 255)
extern "C" __global__ void finish(const float* __restrict__ num, const float* __restrict__ den, const long long n,
                                  float* __restrict__ val, unsigned char* __restrict__ dz) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w = den[i];
    const int q = (int)floorf(fminf(w, 1.0f) * 255.0f + 0.5f);
    dz[i] = (unsigned char)q;
    val[i] = (q > 0 && w > 0.0f) ? num[i] / w : 0.0f;
}

// ---- batched versions for the stack pass: every frame of a strip in one launch ----
// Decoding, finish, feather weight and normalization are the same, expression for expression, as
// decode_gpu (bswap kernels), finish, kernels.feather_w and kernels.prep_px. The drop's corners are
// computed in double-float arithmetic (each value carried as a hi + lo pair of floats, ~48 bits)
// instead of FP64, which GeForce cards run at 1/64 of the float rate. The float coordinates handed to
// sgarea then match the FP64 kernel's except for rare last-bit roundings: differences of ~1e-7, the
// size of those the order of the atomic additions already causes from run to run.
struct dflt { float hi, lo; };
__device__ __forceinline__ dflt df_norm(const float s, const float e) {        // fast two-sum
    const float h = __fadd_rn(s, e);
    return {h, __fsub_rn(e, __fsub_rn(h, s))};
}
__device__ __forceinline__ dflt df_add(const dflt x, const dflt y) {
    const float s = __fadd_rn(x.hi, y.hi);
    const float bb = __fsub_rn(s, x.hi);
    const float err = __fadd_rn(__fsub_rn(x.hi, __fsub_rn(s, bb)), __fsub_rn(y.hi, bb));    // two-sum
    return df_norm(s, __fadd_rn(err, __fadd_rn(x.lo, y.lo)));
}
__device__ __forceinline__ dflt df_mulf(const dflt x, const float y) {          // y: an exact float
    const float p = __fmul_rn(x.hi, y);
    return df_norm(p, __fadd_rn(__fmaf_rn(x.hi, y, -p), __fmul_rn(x.lo, y)));
}
__device__ __forceinline__ bool df_lt(const dflt a, const dflt b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
__device__ __forceinline__ int df_floor(const dflt a) {
    const float f = floorf(a.hi);
    return (int)f - ((f == a.hi && a.lo < 0.0f) ? 1 : 0);
}
__device__ __forceinline__ int df_ceil(const dflt a) {
    const float c = ceilf(a.hi);
    return (int)c + ((c == a.hi && a.lo > 0.0f) ? 1 : 0);
}
__device__ __forceinline__ float df_sub_int(const dflt a, const int u) {        // (float)(a - u)
    return __fadd_rn(__fsub_rn(a.hi, (float)u), a.lo);
}

// blockIdx.y = frame. raw: each frame's band of rows [by0, by0 + bnr) as stored in the file (big-endian;
// bflip = ROWORDER BOTTOM-UP, rows stored in reverse). Drizzles one colour into that frame's own slot of
// the flat num / den buffers (soff, sw columns x snr rows); slots never overlap.
// Threads only visit that colour's pixels: cp0 / cp1 = its column parity in even / odd image rows
// (-1 = none; a Bayer row has at most one), so no lanes idle on other colours; cp0 = -2: no CFA, every
// pixel. G, 28 floats per frame: (hi, lo) of the transform a b c d e f, then of the corner offsets
// (+-a +-b) * half for corners (-,-) (+,-) (+,+) (-,+), then (+-d +-e) * half for the same corners.
extern "C" __global__ void drizzle_batch(const unsigned char* __restrict__ raw, const long long* __restrict__ roff,
                                         const int* __restrict__ by0, const int* __restrict__ bnr,
                                         const int* __restrict__ bflip, const float* __restrict__ bscale,
                                         const float* __restrict__ bzero, const int W, const int bitpix,
                                         const int cp0, const int cp1, const float* __restrict__ G,
                                         const long long* __restrict__ soff, const int* __restrict__ sw,
                                         const int* __restrict__ snr, float* __restrict__ num,
                                         float* __restrict__ den) {
    const int fr = blockIdx.y;
    const int nrows = bnr[fr], ybase = by0[fr];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;        // < W * nrows < 2^31 per frame band
    int r, x;
    if (cp0 == -2) {
        if (i >= W * nrows) return;
        r = i / W;
        x = i - r * W;
    } else {
        const int wh = (W + 1) >> 1;
        const int j = i / wh;
        r = (cp0 >= 0 && cp1 >= 0) ? j : ((((cp0 >= 0) ? 0 : 1) - ybase) & 1) + 2 * j;
        if (r >= nrows) return;
        x = 2 * (i - j * wh) + (((ybase + r) & 1) ? cp1 : cp0);
        if (x >= W) return;
    }
    const int y = ybase + r;
    const long long e = (long long)(bflip[fr] ? nrows - 1 - r : r) * W + x;
    float v;
    if (bitpix == 16) {
        const unsigned short xx = ((const unsigned short*)(raw + roff[fr]))[e];
        const short s = (short)__byte_perm((unsigned int)xx, 0, 0x3201);
        v = ((float)s * bscale[fr] + bzero[fr]) * (1.0f / 65535.0f);
    } else if (bitpix == -32) {
        const unsigned int xx = ((const unsigned int*)(raw + roff[fr]))[e];
        v = __int_as_float(__byte_perm(xx, 0, 0x0123));
    } else {
        v = (float)raw[roff[fr] + e] * (1.0f / 255.0f);
    }
    const float* g = G + 28 * fr;
    const float cx = x + 0.5f, cy = y + 0.5f;                   // exact
    const dflt qcx = df_add(df_add(df_mulf({g[0], g[1]}, cx), df_mulf({g[2], g[3]}, cy)), {g[4], g[5]});
    const dflt qcy = df_add(df_add(df_mulf({g[6], g[7]}, cx), df_mulf({g[8], g[9]}, cy)), {g[10], g[11]});
    dflt qx[4], qy[4];
    for (int k = 0; k < 4; k++) {
        qx[k] = df_add(qcx, {g[12 + 2 * k], g[13 + 2 * k]});
        qy[k] = df_add(qcy, {g[20 + 2 * k], g[21 + 2 * k]});
    }
    dflt minx = qx[0], maxx = qx[0], miny = qy[0], maxy = qy[0];
    for (int k = 1; k < 4; k++) {
        if (df_lt(qx[k], minx)) minx = qx[k];
        if (df_lt(maxx, qx[k])) maxx = qx[k];
        if (df_lt(qy[k], miny)) miny = qy[k];
        if (df_lt(maxy, qy[k])) maxy = qy[k];
    }
    const int ow = sw[fr], oh = snr[fr];
    const int u0 = max(df_floor(minx), 0), u1 = min(df_ceil(maxx) - 1, ow - 1);
    const int v0 = max(df_floor(miny), 0), v1 = min(df_ceil(maxy) - 1, oh - 1);
    if (u0 > u1 || v0 > v1) return;
    float* nump = num + soff[fr];
    float* denp = den + soff[fr];
    for (int vv = v0; vv <= v1; vv++) {
        float ly[4];
        for (int k = 0; k < 4; k++) ly[k] = df_sub_int(qy[k], vv);
        for (int uu = u0; uu <= u1; uu++) {
            float lx[4];
            for (int k = 0; k < 4; k++) lx[k] = df_sub_int(qx[k], uu);
            float s = 0.0f;
            for (int k = 0; k < 4; k++) {
                const int k2 = (k + 1) & 3;
                s += sgarea(lx[k], ly[k], lx[k2], ly[k2]);
            }
            const float area = fabsf(s);
            if (area > 0.0f) {
                const long long o = (long long)vv * ow + uu;
                atomicAdd(nump + o, area * v);
                atomicAdd(denp + o, area);
            }
        }
    }
}

// In place over the slots [base, base + n) of frames soff[0..nf): num/den -> finish -> feather weight
// -> addscale normalization, leaving the stack value (NaN = no data) in vals and its weight in wts.
extern "C" __global__ void finish_prep_batch(float* __restrict__ vals, float* __restrict__ wts,
                                             const long long base, const long long n,
                                             const long long* __restrict__ soff, const int nf,
                                             const int* __restrict__ sw, const int* __restrict__ sra,
                                             const double* __restrict__ A, const double* __restrict__ OFF,
                                             const int feather, const unsigned long long* __restrict__ mptr,
                                             const int* __restrict__ mh, const int* __restrict__ mw,
                                             const float* __restrict__ fy, const float* __restrict__ fx,
                                             const float* __restrict__ ramp, const float dist,
                                             const float invdist) {
    const long long i = base + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= base + n) return;
    int lo = 0, hi = nf - 1;                        // the last frame whose slot starts at or before i
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (soff[mid] <= i) lo = mid; else hi = mid - 1;
    }
    const int fr = lo;
    const int w = sw[fr];
    const int local = (int)(i - soff[fr]);                  // within one frame's slot: < 2^31
    const int row = local / w, col = local - row * w;
    // finish
    const float den_ = wts[i];
    const int q = (int)floorf(fminf(den_, 1.0f) * 255.0f + 0.5f);
    const unsigned char dz = (unsigned char)q;
    const float v = (q > 0 && den_ > 0.0f) ? vals[i] / den_ : 0.0f;
    // feather weight
    float fw = 1.0f;
    if (feather) {
        const float* m = (const float*)mptr[fr];
        const int MH = mh[fr], MW = mw[fr];
        const int rr = sra[fr] + row, cc = col;
        float yy = fminf(fmaxf((rr + 0.5f) * fy[fr] - 0.5f, 0.0f), (float)(MH - 1));
        float xx = fminf(fmaxf((cc + 0.5f) * fx[fr] - 0.5f, 0.0f), (float)(MW - 1));
        const int y0 = (int)yy, x0 = (int)xx, y1 = min(y0 + 1, MH - 1), x1 = min(x0 + 1, MW - 1);
        const float dy = yy - y0, dx = xx - x0;
        const float d = (m[y0 * MW + x0] * (1.0f - dx) + m[y0 * MW + x1] * dx) * (1.0f - dy)
                      + (m[y1 * MW + x0] * (1.0f - dx) + m[y1 * MW + x1] * dx) * dy;
        if (d == 0.0f) fw = 0.0f;
        else if (d > dist) fw = 1.0f;
        else {
            int idx = (int)(__fmul_rn(__fmul_rn(d, invdist), 1000.0f));
            idx = max(0, min(idx, 1000));
            fw = ramp[idx];
        }
    }
    // normalize (prep_px with dscale 1, dzero 0)
    const float nanf_ = __int_as_float(0x7fc00000);
    const float dd = (float)dz * 1.0f + 0.0f;
    if (v == 0.0f || dd == 0.0f || isnan(v)) { vals[i] = nanf_; wts[i] = 0.0f; }
    else {
        const float nv = (float)__dsub_rn(__dmul_rn((double)v, A[fr]), OFF[fr]);
        if (nv == 0.0f) { vals[i] = nanf_; wts[i] = 0.0f; }
        else { vals[i] = nv; wts[i] = dd * fw; }
    }
}
"""
_mod = None


def _fn(name: str):
    global _mod
    if _mod is None:
        import cupy as cp
        _mod = cp.RawModule(code=_SRC)
    return _mod.get_function(name)


def df_split(v) -> np.ndarray:
    """float64 values -> float32 (hi, lo) pairs on a new last axis; hi + lo carries ~48 bits of each."""
    v = np.asarray(v, dtype=np.float64)
    hi = v.astype(np.float32)
    return np.stack([hi, (v - hi.astype(np.float64)).astype(np.float32)], axis=-1)


_CORNERS = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])     # drizzle's corner order


def batch_geometry(T, pixfrac: float) -> np.ndarray:
    """(n, 6) transforms (frame -> slot coordinates) -> the (n, 28) float32 table of drizzle_batch."""
    T = np.asarray(T, dtype=np.float64).reshape(-1, 6)
    half = pixfrac / 2.0
    ox = (_CORNERS[:, 0] * T[:, 0:1] + _CORNERS[:, 1] * T[:, 1:2]) * half
    oy = (_CORNERS[:, 0] * T[:, 3:4] + _CORNERS[:, 1] * T[:, 4:5]) * half
    n = len(T)
    return np.concatenate([df_split(T).reshape(n, 12), df_split(ox).reshape(n, 8),
                           df_split(oy).reshape(n, 8)], axis=1)


def site_parities(lut, ch: int) -> tuple[int, int] | None:
    """Column parity of colour ch in the even and the odd rows of a 2x2 CFA (-1: none there), or None
    when a row holds it twice (not a Bayer layout)."""
    out = []
    for row in np.asarray(lut).reshape(2, 2):
        cols = [cx for cx in (0, 1) if int(row[cx]) == ch]
        if len(cols) > 1:
            return None
        out.append(cols[0] if cols else -1)
    return out[0], out[1]


def frame_bbox(t, width: int, height: int, pixfrac: float = 1.0) -> tuple[float, float, float, float]:
    """Output-coordinate bounding box (xmin, ymin, xmax, ymax) of a frame's drizzled footprint.
    t: 2x3 affine mapping input continuous coords to output continuous coords."""
    m = np.asarray(t, dtype=np.float64).reshape(2, 3)
    pad = 0.5 - pixfrac / 2.0
    xs = np.array([pad, width - pad, width - pad, pad])
    ys = np.array([pad, pad, height - pad, height - pad])
    ox = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
    oy = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
    return float(ox.min()), float(oy.min()), float(ox.max()), float(oy.max())


def drizzle_into(img, y0: int, t, lut, pixfrac: float, num, den, only_ch: int = -1) -> None:
    """Accumulate rows [y0, y0 + img.shape[0]) of a frame into num/den (planes, oh, ow).

    img: CuPy float32 (nrows, W), top-down. t: 2x3 affine from frame coords to *buffer* coords
    (i.e. already shifted so the buffer's top-left is (0, 0)). lut: (2, 2) CFA channels or None."""
    nrows, w = img.shape
    oh, ow = num.shape[-2], num.shape[-1]
    m = np.asarray(t, dtype=np.float64).reshape(2, 3)
    cfa = lut is not None
    l = np.asarray(lut).ravel() if cfa else np.zeros(4, int)
    n = nrows * w
    if n == 0:
        return
    _fn("drizzle")(((n + 255) // 256,), (256,), (
        img, np.int32(w), np.int32(nrows), np.int32(y0), np.int32(1 if cfa else 0),
        np.int32(l[0]), np.int32(l[1]), np.int32(l[2]), np.int32(l[3]), np.int32(only_ch),
        np.float64(m[0, 0]), np.float64(m[0, 1]), np.float64(m[0, 2]),
        np.float64(m[1, 0]), np.float64(m[1, 1]), np.float64(m[1, 2]),
        np.float64(pixfrac / 2.0), np.int32(ow), np.int32(oh), num, den))


def finish(num, den):
    """-> (values float32, uint8 weights) with the same shape."""
    import cupy as cp
    val = cp.empty(num.shape, cp.float32)
    dz = cp.empty(num.shape, cp.uint8)
    n = num.size
    _fn("finish")(((n + 255) // 256,), (256,), (num, den, np.int64(n), val, dz))
    return val, dz


def warp_frame(planes, t, lut, pixfrac: float, out_w: int, out_h: int):
    """Whole frame -> (values (3 or C, out_h, out_w), uint8 weights) in a buffer of the given size.
    planes: CuPy (C, H, W) top-down decoded frame; t: frame -> buffer coords."""
    import cupy as cp
    c = planes.shape[0]
    nplanes = 3 if lut is not None else c
    num = cp.zeros((nplanes, out_h, out_w), cp.float32)
    den = cp.zeros((nplanes, out_h, out_w), cp.float32)
    if lut is not None:
        drizzle_into(cp.ascontiguousarray(planes[0]), 0, t, lut, pixfrac, num, den)
    else:
        for p in range(c):
            drizzle_into(cp.ascontiguousarray(planes[p]), 0, t, None, pixfrac, num[p:p + 1], den[p:p + 1])
    return finish(num, den)


def bbox_int(box) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) integer pixel box containing a float bbox."""
    x0, y0 = math.floor(box[0] + 1e-9), math.floor(box[1] + 1e-9)
    x1, y1 = math.ceil(box[2] - 1e-9), math.ceil(box[3] - 1e-9)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)
