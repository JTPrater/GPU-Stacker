# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
# Contains code ported from Siril and CFITSIO; see THIRD-PARTY-NOTICES.txt
"""CUDA kernels (CuPy). Each one mirrors a specific piece of Siril 1.4.4; see the comments."""
from __future__ import annotations

import os

import cupy as cp
import numpy as np

REJECT_NONE, REJECT_SIGMA, REJECT_WINSORIZED = 0, 1, 2

# ----------------------------------------------------------------------------- decoding

bswap_f32 = cp.ElementwiseKernel(
    "uint32 x", "float32 y", "y = __int_as_float(__byte_perm(x, 0, 0x0123))", "bswap_f32")

# 16-bit FITS (BITPIX 16, usually BZERO 32768) -> float in [0, 1] like Siril's ushort scaling.
bswap_i16 = cp.ElementwiseKernel(
    "uint16 x, float32 bscale, float32 bzero", "float32 y",
    "short s = (short)__byte_perm((unsigned int)x, 0, 0x3201); y = ((float)s * bscale + bzero) * (1.0f / 65535.0f)",
    "bswap_i16")

# ----------------------------------------------------------------------------- per-frame preparation

# Siril ADDITIVE_SCALING: tmp = pixel * pscale (double); stack = (float)(tmp - poffset).
# Null pixels (raw 0, normalized 0, or drizzle weight 0) become NaN = "no data".
prep_px = cp.ElementwiseKernel(
    "float32 v, uint8 dz, float32 fw, float64 a, float64 off, float32 dscale, float32 dzero",
    "float32 val, float32 wt",
    """
    const float nanf_ = __int_as_float(0x7fc00000);
    const float d = (float)dz * dscale + dzero;
    if (v == 0.0f || d == 0.0f || isnan(v)) { val = nanf_; wt = 0.0f; }
    else {
        const float nv = (float)__dsub_rn(__dmul_rn((double)v, a), off);
        if (nv == 0.0f) { val = nanf_; wt = 0.0f; }
        else { val = nv; wt = d * fw; }
    }
    """, "prep_px")

prep_px_nodz = cp.ElementwiseKernel(
    "float32 v, float32 fw, float64 a, float64 off", "float32 val, float32 wt",
    """
    const float nanf_ = __int_as_float(0x7fc00000);
    if (v == 0.0f || isnan(v)) { val = nanf_; wt = 0.0f; }
    else {
        const float nv = (float)__dsub_rn(__dmul_rn((double)v, a), off);
        if (nv == 0.0f) { val = nanf_; wt = 0.0f; }
        else { val = nv; wt = fw; }
    }
    """, "prep_px_nodz")


def ramp_table() -> np.ndarray:
    """Siril's init_ramp(): smootherstep sampled on RAMP_PACE + 1 = 1001 points, in float32."""
    i = np.arange(1001, dtype=np.float32)
    r = i * np.float32(1.0 / 1000.0)
    six, fifteen, ten = np.float32(6), np.float32(15), np.float32(10)
    return (r * r * r * (six * r * r - fifteen * r + ten)).astype(np.float32)


# Feather weight (Siril median_and_mean.c): the 10%-scale distance mask is upscaled with
# INTER_LINEAR; distance 0 -> 0, > feather -> 1, else ramp[(int)(d / feather * 1000)].
# Distances stay in mask pixels, exactly as Siril compares them with -feather.
feather_w = cp.ElementwiseKernel(
    "raw float32 m, raw float32 ramp, int32 mh, int32 mw, int32 w, int32 r0, float32 fy, float32 fx, "
    "float32 dist, float32 invdist",
    "float32 out",
    """
    const int r = r0 + (int)(i / w), c = (int)(i % w);
    float y = fminf(fmaxf((r + 0.5f) * fy - 0.5f, 0.0f), (float)(mh - 1));
    float x = fminf(fmaxf((c + 0.5f) * fx - 0.5f, 0.0f), (float)(mw - 1));
    const int y0 = (int)y, x0 = (int)x, y1 = min(y0 + 1, mh - 1), x1 = min(x0 + 1, mw - 1);
    const float dy = y - y0, dx = x - x0;
    const float d = (m[y0 * mw + x0] * (1.0f - dx) + m[y0 * mw + x1] * dx) * (1.0f - dy)
                  + (m[y1 * mw + x0] * (1.0f - dx) + m[y1 * mw + x1] * dx) * dy;
    if (d == 0.0f) out = 0.0f;
    else if (d > dist) out = 1.0f;
    else {
        int idx = (int)(__fmul_rn(__fmul_rn(d, invdist), 1000.0f));
        idx = max(0, min(idx, 1000));
        out = ramp[idx];
    }
    """, "feather_w")

# ----------------------------------------------------------------------------- column sort

# Sort every pixel's column at once: (pixel << 32 | order-preserving float bits) as one
# 1-D uint64 radix sort, then unpack back to a (K, P) layout. NaN sorts last.
_mk_keys = cp.ElementwiseKernel(
    "float32 v, int64 P", "uint64 key",
    """
    unsigned int u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    key = ((unsigned long long)(i % P) << 32) | u;
    """, "mk_keys")

_unkey = cp.ElementwiseKernel(
    "raw uint64 keys, int64 P, int64 K", "float32 s",
    """
    const long long r = i / P, p = i % P;
    unsigned int u = (unsigned int)(keys[p * K + r] & 0xffffffffull);
    u = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
    s = __uint_as_float(u);
    """, "unkey")


def sort_columns_radix(V2: cp.ndarray) -> cp.ndarray:
    """Sort a (K, P) float32 array along K (NaN last) with a single radix sort."""
    K, P = V2.shape
    keys = _mk_keys(V2, np.int64(P)).ravel()
    keys.sort()
    S = cp.empty((K, P), dtype=cp.float32)
    _unkey(keys, np.int64(P), np.int64(K), S)
    return S


# Per-pixel bitonic sort in shared memory: the same order-preserving 32-bit keys as the radix sort
# (so the same order, NaN last), but each pixel's K values are sorted on chip by one row of threads
# instead of one radix sort over K * P keys. Padding keys 0xffffffff sort after everything.
_BITONIC_SRC = r"""
__device__ __forceinline__ unsigned int to_key(float v) {
    unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float from_key(unsigned int u) {
    u = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
    return __uint_as_float(u);
}
// block = (TX, PPB): row y of PPB sorts pixel p0 + y; KP = power of two >= K; shared: PPB * KP keys.
extern "C" __global__ void sort_columns_bitonic(const float* __restrict__ V, float* __restrict__ S,
                                                const int K, const long long P, const int KP) {
    extern __shared__ unsigned int sh[];
    const int TX = blockDim.x, PPB = blockDim.y;
    const int lin = threadIdx.y * TX + threadIdx.x, nthr = TX * PPB;
    const long long p0 = (long long)blockIdx.x * PPB;
    // load: consecutive threads take consecutive pixels of one row k (coalesced)
    for (int e = lin; e < KP * PPB; e += nthr) {
        const int k = e / PPB, y = e - k * PPB;
        const long long p = p0 + y;
        unsigned int key = 0xffffffffu;
        if (k < K && p < P) key = to_key(V[(long long)k * P + p]);
        sh[y * KP + k] = key;
    }
    __syncthreads();
    unsigned int* s = sh + threadIdx.y * KP;
    for (int size = 2; size <= KP; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = threadIdx.x; t < (KP >> 1); t += TX) {
                const int i = 2 * t - (t & (stride - 1));       // first index of the pair
                const int j = i + stride;
                const bool up = (i & size) == 0;
                const unsigned int a = s[i], b = s[j];
                if ((a > b) == up) { s[i] = b; s[j] = a; }
            }
            __syncthreads();
        }
    }
    for (int e = lin; e < KP * PPB; e += nthr) {
        const int k = e / PPB, y = e - k * PPB;
        const long long p = p0 + y;
        if (k < K && p < P) S[(long long)k * P + p] = from_key(sh[y * KP + k]);
    }
}
"""
_bitonic = cp.RawModule(code=_BITONIC_SRC).get_function("sort_columns_bitonic")
BITONIC_SHARED = 32 * 1024            # bytes of shared memory per block


def sort_columns(V2: cp.ndarray) -> cp.ndarray:
    """Sort a (K, P) float32 array along K, NaN last. Per-pixel bitonic sort in shared memory when a
    column fits (K <= 8192), else one radix sort; both give the identical result."""
    K, P = V2.shape
    KP = 1 << max(1, (K - 1).bit_length())
    if KP * 4 > BITONIC_SHARED or os.environ.get("GPUSTACKER_RADIX_SORT") == "1":   # env: for comparisons
        return sort_columns_radix(V2)
    tx = min(KP // 2, 64)                                    # threads per pixel (measured best on an RTX 3060)
    ppb = max(1, min(BITONIC_SHARED // (KP * 4), 256 // tx))    # pixels per block
    S = cp.empty((K, P), dtype=cp.float32)
    V2 = cp.ascontiguousarray(V2)
    _bitonic(((P + ppb - 1) // ppb,), (tx, ppb), (V2, S, np.int32(K), np.int64(P), np.int32(KP)),
             shared_mem=ppb * KP * 4)
    return S


# ----------------------------------------------------------------------------- rejection, mean

_SOURCE = r"""
#define F_INF __int_as_float(0x7f800000)

// siril_stats_float_sd of S[lo:hi] (column p) after clamping to [b0, b1]: double sums, float mean,
// float squared deviations, N-1. The _rn intrinsics stop FMA contraction (Siril's x86 build has none).
__device__ float sd_clamped(const float* __restrict__ S, long long P, long long p,
                            int lo, int hi, float b0, float b1) {
    const int N = hi - lo;
    double acc = 0.0;
    for (int k = lo; k < hi; ++k) acc = __dadd_rn(acc, (double)fminf(fmaxf(S[k * P + p], b0), b1));
    const float mean = (float)__ddiv_rn(acc, (double)N);
    acc = 0.0;
    for (int k = lo; k < hi; ++k) {
        const float d = __fsub_rn(fminf(fmaxf(S[k * P + p], b0), b1), mean);
        acc = __dadd_rn(acc, (double)__fmul_rn(d, d));
    }
    return sqrtf((float)__ddiv_rn(acc, (double)(N - 1)));
}

__device__ __forceinline__ float median_sorted(const float* __restrict__ S, long long P, long long p,
                                               int lo, int hi) {
    const int N = hi - lo, k = N / 2;
    return (N & 1) ? S[(long long)(lo + k) * P + p]
                   : __fmul_rn(0.5f, __fadd_rn(S[(long long)(lo + k - 1) * P + p], S[(long long)(lo + k) * P + p]));
}

// Siril apply_rejection_float on per-pixel sorted columns (NaN = no data, sorted last).
// mode 0 = none, 1 = sigma clipping, 2 = winsorized sigma clipping.
// Siril's "keep at least 4" guard: a pass may reject while N - r > 4, with r counting rejections
// over all passes (never reset). When the guard cuts a pass short, which outliers go depends on the
// order Siril visits its stack in: frame order, rearranged in place by every median. Below 9 values
// its median is a sorting network, so the stack is sorted: the lowest outliers go first, then the
// smallest high ones, as computed here. From 9 values on, quickselect leaves an order that only the
// frame order reproduces: the pixel goes on the `bind` list and reject_fixup redoes it (with no budget
// left nothing goes, in any order). No later pass can reject anything after the guard binds, so the
// pixel is final; its kept set may then have a gap.
extern "C" __global__ void reject(const float* __restrict__ S, const int K, const long long P,
                                  const int mode, const float siglow, const float sighigh,
                                  float* __restrict__ pmin, float* __restrict__ pmax,
                                  int* __restrict__ kept, int* __restrict__ nvalid,
                                  int* __restrict__ nlo, int* __restrict__ nhi,
                                  float* __restrict__ kmean, int* __restrict__ nbind, int* __restrict__ bind) {
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int lo = 0, hi = K;
    while (lo < hi) {                       // count valid values
        const int mid = (lo + hi) >> 1;
        if (isnan(S[(long long)mid * P + p])) hi = mid; else lo = mid + 1;
    }
    const int n = lo;
    nvalid[p] = n;
    if (n == 0) { kept[p] = 0; pmin[p] = 1.f; pmax[p] = 0.f; nlo[p] = 0; nhi[p] = 0; kmean[p] = 0.f; return; }
    lo = 0; hi = n;
    int rtot = 0, tlo = 0, thi = 0;
    bool changed = (mode != 0) && n > 1;
    while (changed && hi - lo > 3) {
        const int N = hi - lo;
        const float median = median_sorted(S, P, p, lo, hi);
        float sigma = sd_clamped(S, P, p, lo, hi, -F_INF, F_INF);
        if (mode == 2) {
            float sig_clamp = sigma, sigma0;
            int it = 0;
            do {                            // in-place clamping == clamp at the smallest sigma so far
                sig_clamp = fminf(sig_clamp, sigma);
                sigma0 = sigma;
                const float h = __fmul_rn(1.5f, sig_clamp);
                sigma = __fmul_rn(1.134f, sd_clamped(S, P, p, lo, hi, __fsub_rn(median, h), __fadd_rn(median, h)));
            } while (fabsf(__fsub_rn(sigma, sigma0)) > __fmul_rn(sigma0, 0.0005f) && ++it < 1000);
        }
        const float tl = __fmul_rn(sigma, siglow), th = __fmul_rn(sigma, sighigh);
        int rl = 0, rh = 0;
        while (lo + rl < hi && __fsub_rn(median, S[(long long)(lo + rl) * P + p]) > tl) ++rl;
        while (hi - 1 - rh >= lo + rl && __fsub_rn(S[(long long)(hi - 1 - rh) * P + p], median) > th) ++rh;
        const int budget = max(0, N - rtot - 4);
        if (rl + rh > budget) {             // guard binds: kept = S[lo+nl, hi-rh) + S[hi-rh+nh, hi)
            if (budget > 0 && N >= 9) { bind[atomicAdd(nbind, 1)] = (int)p; return; }
            const int nl = min(rl, budget), nh = budget - nl;
            const int a0 = lo + nl, g0 = hi - rh, g1 = hi - rh + nh;
            double s = 0.0;
            for (int k = a0; k < hi; ++k)
                if (k < g0 || k >= g1) s = __dadd_rn(s, (double)S[(long long)k * P + p]);
            const int nk = N - budget;
            kept[p] = nk;
            nlo[p] = tlo + nl; nhi[p] = thi + nh;
            pmin[p] = S[(long long)a0 * P + p];
            pmax[p] = S[(long long)((nh < rh) ? hi - 1 : g0 - 1) * P + p];
            kmean[p] = (float)(s / nk);
            return;
        }
        lo += rl; hi -= rh;
        rtot += rl + rh; tlo += rl; thi += rh;
        changed = (rl + rh) > 0;
    }
    const int nk = hi - lo;
    kept[p] = nk;
    nlo[p] = tlo; nhi[p] = thi;
    pmin[p] = S[(long long)lo * P + p];
    pmax[p] = S[(long long)(hi - 1) * P + p];
    double s = 0.0;                         // Siril: double sum of the kept values / kept
    for (int k = lo; k < hi; ++k) s = __dadd_rn(s, (double)S[(long long)k * P + p]);
    kmean[p] = (float)(s / nk);
}

// ---- the same rejection with one warp per pixel (reject_w) ----
// Each block takes blockDim.x / 32 consecutive pixels and copies their sorted columns to shared memory
// once; each warp then makes its pixel's passes over its own column there, the lanes taking strided
// slices. The thread-per-pixel kernel above streams every column from global memory twice per pass
// (some 90 times a pixel in winsorized clipping) and adds in FP64, which GeForce cards run at 1/64 rate.
// Here Siril's double sums of floats are computed exactly, in 64-bit fixed point: integer adds, in any
// order. A double holds the exact sum of floats whose sizes span less than ~2^20, so for the sums of
// values the result is the same double. Sums of squared deviations can hold tiny terms whose low bits the
// double drops; there the two doubles may differ by an ulp, which changes the float made from them only
// in the rarest ties. Tests check the two kernels give identical bits, as they do on the M45 data.
__device__ __forceinline__ int fx_scale(const float m, const int n) {
    // units of 2^s such that n terms of magnitude <= m sum within 63 bits: m < 2^e, n < 2^nb
    const int e = (int)((__float_as_uint(m) >> 23) & 0xffu) - 126;
    return e + (32 - __clz(n)) - 63;
}
__device__ __forceinline__ long long fx(const float v, const int s) {      // round(v / 2^s), half to even
    const unsigned int u = __float_as_uint(v);
    const int ex = (int)((u >> 23) & 0xffu);
    if (ex == 0) return 0;                  // zero (subnormals are far below any scale used here)
    const long long m = (long long)((u & 0x7fffffu) | 0x800000u);
    const int sh = ex - 150 - s;            // |v| = m * 2^(ex - 150); sh <= 39 by the choice of s
    long long q;
    if (sh >= 0) q = m << sh;
    else if (sh > -40) {
        const int r = -sh;
        const long long half = 1ll << (r - 1), rem = m & ((1ll << r) - 1);
        q = m >> r;
        if (rem > half || (rem == half && (q & 1))) ++q;
    } else q = 0;
    return (u >> 31) ? -q : q;
}
__device__ __forceinline__ double fx_sum(long long q, const int s) {       // warp total as a double
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    return ldexp(__ll2double_rn(q), s);
}

// sd_clamped on a shared-memory column; every lane returns the same value
__device__ float sd_clamped_w(const float* __restrict__ c, const int lo, const int hi, const float b0,
                              const float b1, const int lane) {
    const int N = hi - lo;
    const float vlo = fminf(fmaxf(c[lo], b0), b1), vhi = fminf(fmaxf(c[hi - 1], b0), b1);
    int s = fx_scale(fmaxf(fabsf(vlo), fabsf(vhi)), N);
    long long q = 0;
    for (int k = lo + lane; k < hi; k += 32) q += fx(fminf(fmaxf(c[k], b0), b1), s);
    const float mean = (float)__ddiv_rn(fx_sum(q, s), (double)N);
    const float dlo = __fsub_rn(vlo, mean), dhi = __fsub_rn(vhi, mean);    // sorted: the extremes
    s = fx_scale(fmaxf(__fmul_rn(dlo, dlo), __fmul_rn(dhi, dhi)), N);
    q = 0;
    for (int k = lo + lane; k < hi; k += 32) {
        const float d = __fsub_rn(fminf(fmaxf(c[k], b0), b1), mean);
        q += fx(__fmul_rn(d, d), s);
    }
    return sqrtf((float)__ddiv_rn(fx_sum(q, s), (double)(N - 1)));
}

extern "C" __global__ void reject_w(const float* __restrict__ S, const int K, const long long P, const int kpad,
                                    const int mode, const float siglow, const float sighigh,
                                    float* __restrict__ pmin, float* __restrict__ pmax,
                                    int* __restrict__ kept, int* __restrict__ nvalid,
                                    int* __restrict__ nlo, int* __restrict__ nhi,
                                    float* __restrict__ kmean, int* __restrict__ nbind, int* __restrict__ bind) {
    extern __shared__ float col[];
    const int ppb = blockDim.x >> 5;
    const long long p0 = (long long)blockIdx.x * ppb;
    for (int i = threadIdx.x; i < K * ppb; i += blockDim.x) {
        const int k = i / ppb, j = i - k * ppb;          // consecutive threads: consecutive pixels of a row
        col[j * kpad + k] = (p0 + j < P) ? S[(long long)k * P + p0 + j] : __int_as_float(0x7fc00000);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wj = threadIdx.x >> 5;
    const long long p = p0 + wj;
    if (p >= P) return;
    const float* __restrict__ c = col + wj * kpad;
    int lo = 0, hi = K;
    while (lo < hi) {                       // count valid values
        const int mid = (lo + hi) >> 1;
        if (isnan(c[mid])) hi = mid; else lo = mid + 1;
    }
    const int n = lo;
    if (n == 0) {
        if (lane == 0) { nvalid[p] = 0; kept[p] = 0; pmin[p] = 1.f; pmax[p] = 0.f; nlo[p] = 0; nhi[p] = 0; kmean[p] = 0.f; }
        return;
    }
    lo = 0; hi = n;
    int rtot = 0, tlo = 0, thi = 0;
    bool changed = (mode != 0) && n > 1;
    while (changed && hi - lo > 3) {
        const int N = hi - lo, h2 = lo + N / 2;
        const float median = (N & 1) ? c[h2] : __fmul_rn(0.5f, __fadd_rn(c[h2 - 1], c[h2]));
        float sigma = sd_clamped_w(c, lo, hi, -F_INF, F_INF, lane);
        if (mode == 2) {
            float sig_clamp = sigma, sigma0;
            int it = 0;
            do {
                sig_clamp = fminf(sig_clamp, sigma);
                sigma0 = sigma;
                const float h = __fmul_rn(1.5f, sig_clamp);
                sigma = __fmul_rn(1.134f, sd_clamped_w(c, lo, hi, __fsub_rn(median, h), __fadd_rn(median, h), lane));
            } while (fabsf(__fsub_rn(sigma, sigma0)) > __fmul_rn(sigma0, 0.0005f) && ++it < 1000);
        }
        const float tl = __fmul_rn(sigma, siglow), th = __fmul_rn(sigma, sighigh);
        int rl = 0, rh = 0;
        while (lo + rl < hi && __fsub_rn(median, c[lo + rl]) > tl) ++rl;
        while (hi - 1 - rh >= lo + rl && __fsub_rn(c[hi - 1 - rh], median) > th) ++rh;
        const int budget = max(0, N - rtot - 4);
        if (rl + rh > budget) {             // the keep-4 guard binds (see reject)
            if (budget > 0 && N >= 9) {
                if (lane == 0) bind[atomicAdd(nbind, 1)] = (int)p;
                return;
            }
            const int nl = min(rl, budget), nh = budget - nl;
            const int a0 = lo + nl, g0 = hi - rh, g1 = hi - rh + nh;
            const int s = fx_scale(fmaxf(fabsf(c[a0]), fabsf(c[hi - 1])), hi - a0);
            long long q = 0;
            for (int k = a0 + lane; k < hi; k += 32)
                if (k < g0 || k >= g1) q += fx(c[k], s);
            const double sum = fx_sum(q, s);
            const int nk = N - budget;
            if (lane == 0) {
                nvalid[p] = n; kept[p] = nk;
                nlo[p] = tlo + nl; nhi[p] = thi + nh;
                pmin[p] = c[a0];
                pmax[p] = c[(nh < rh) ? hi - 1 : g0 - 1];
                kmean[p] = (float)(sum / nk);
            }
            return;
        }
        lo += rl; hi -= rh;
        rtot += rl + rh; tlo += rl; thi += rh;
        changed = (rl + rh) > 0;
    }
    const int nk = hi - lo;
    const int s = fx_scale(fmaxf(fabsf(c[lo]), fabsf(c[hi - 1])), nk);
    long long q = 0;
    for (int k = lo + lane; k < hi; k += 32) q += fx(c[k], s);
    const double sum = fx_sum(q, s);
    if (lane == 0) {
        nvalid[p] = n; kept[p] = nk; nlo[p] = tlo; nhi[p] = thi;
        pmin[p] = c[lo]; pmax[p] = c[hi - 1];
        kmean[p] = (float)(sum / nk);
    }
}

// ---- the pixels where the keep-4 guard binds at 9+ values (rare): Siril's steps in Siril's order ----
// quickmedian_float (algos/sorting.c): sorting networks below 9 values, else quickselect; both rearrange
// a in place, and that order is the one the guard visits.
#define SW(i, j) if (a[i] > a[j]) { const float t_ = a[i]; a[i] = a[j]; a[j] = t_; }
__device__ double quickmedian_f(float* __restrict__ a, const int n) {
    const int k = n / 2;
    if (n < 9) {
        switch (n) {
            case 1: return (double)a[0];
            case 2: SW(0, 1); break;
            case 3: SW(0, 1); SW(1, 2); SW(0, 1); break;
            case 4: SW(0, 1); SW(2, 3); SW(0, 2); SW(1, 3); SW(1, 2); break;
            case 5: SW(0, 1); SW(2, 3); SW(1, 3); SW(2, 4); SW(0, 2); SW(1, 4); SW(1, 2); SW(3, 4); SW(2, 3); break;
            case 6: SW(0, 1); SW(2, 3); SW(4, 5); SW(0, 2); SW(3, 5); SW(1, 4); SW(0, 1); SW(2, 3); SW(4, 5);
                    SW(1, 2); SW(3, 4); SW(2, 3); break;
            case 7: SW(1, 2); SW(3, 4); SW(5, 6); SW(0, 2); SW(4, 6); SW(3, 5); SW(2, 6); SW(1, 5); SW(0, 4);
                    SW(2, 5); SW(0, 3); SW(2, 4); SW(1, 3); SW(0, 1); SW(2, 3); SW(4, 5); break;
            case 8: SW(0, 1); SW(2, 3); SW(4, 5); SW(6, 7); SW(0, 2); SW(1, 3); SW(4, 6); SW(5, 7); SW(1, 2);
                    SW(5, 6); SW(0, 4); SW(1, 5); SW(2, 6); SW(3, 7); SW(2, 4); SW(3, 5); SW(1, 2); SW(3, 4);
                    SW(5, 6); break;
            default: return 0.0;
        }
        return (n & 1) ? (double)a[k] : __dmul_rn((double)__fadd_rn(a[k - 1], a[k]), 0.5);   // float sum
    }
    int left = 0, right = n - 1;
    while (left < right) {
        int pindex = (left + right) / 2;
        const float pivot = a[pindex];
        a[pindex] = a[right];
        a[right] = pivot;
        pindex = left;
        for (int i = left; i < right; ++i)
            if (a[i] < pivot) { const float t = a[pindex]; a[pindex] = a[i]; a[i] = t; ++pindex; }
        a[right] = a[pindex];
        a[pindex] = pivot;
        if (pindex < k) left = pindex + 1; else right = pindex;
    }
    return (n & 1) ? (double)a[k] : __dmul_rn(__dadd_rn((double)a[k - 1], (double)a[k]), 0.5);    // double sum
}
#undef SW

// apply_rejection_float as written, for the pixels reject / reject_w put on the bind list: the valid values
// in frame order (V, NaN = no data), every median rearranging them in place, the guard visiting that order,
// each pass compacting what it keeps; then mean_and_reject's min, max and double sum of the kept values.
// A thread per listed pixel, on its own K floats of `work`. (Siril also gives up on a sigma-clipped pixel
// whose first median is exactly 0; like the kernels above, this does not.)
extern "C" __global__ void reject_fixup(const float* __restrict__ V, const int K, const long long P,
                                        const int mode, const float siglow, const float sighigh,
                                        const int* __restrict__ nbind, const int* __restrict__ bind,
                                        float* __restrict__ work,
                                        float* __restrict__ pmin, float* __restrict__ pmax,
                                        int* __restrict__ kept, int* __restrict__ nvalid,
                                        int* __restrict__ nlo, int* __restrict__ nhi,
                                        float* __restrict__ kmean) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* __restrict__ a = work + (long long)tid * K;
    const int nb = *nbind;
    for (int t = tid; t < nb; t += gridDim.x * blockDim.x) {
        const long long p = bind[t];
        int N = 0;
        for (int k = 0; k < K; ++k) {
            const float v = V[(long long)k * P + p];
            if (!isnan(v)) a[N++] = v;
        }
        nvalid[p] = N;
        int r = 0, rlo = 0, rhi = 0;
        double median = (mode == 1) ? quickmedian_f(a, N) : 0.0;    // sigma clipping: before the first pass
        bool first = true, changed;
        do {
            float sigma = sd_clamped(a, 1, 0, 0, N, -F_INF, F_INF);
            if (mode == 2 || !first) median = quickmedian_f(a, N);
            first = false;
            const float m = (float)median;
            if (mode == 2) {
                float sig_clamp = sigma, sigma0;
                int it = 0;
                do {                            // w_stack clamped in place == clamped at the smallest sigma so far
                    sig_clamp = fminf(sig_clamp, sigma);
                    sigma0 = sigma;
                    const float h = __fmul_rn(1.5f, sig_clamp);
                    sigma = __fmul_rn(1.134f, sd_clamped(a, 1, 0, 0, N, __fsub_rn(m, h), __fadd_rn(m, h)));
                } while (fabsf(__fsub_rn(sigma, sigma0)) > __fmul_rn(sigma0, 0.0005f) && ++it < 1000);
            }
            const float tl = __fmul_rn(sigma, siglow), th = __fmul_rn(sigma, sighigh);
            int out = 0;
            for (int i = 0; i < N; ++i) {       // visit in the current order, compacting as we go
                const float v = a[i];
                bool rej = false;
                if (N - r > 4) {
                    if (__fsub_rn(m, v) > tl) { rej = true; ++rlo; }
                    else if (__fsub_rn(v, m) > th) { rej = true; ++rhi; }
                    r += rej;
                }
                if (!rej) a[out++] = v;
            }
            changed = out != N;
            N = out;
        } while (changed && N > 3);
        float lo = a[0], hi = a[0];
        double s = 0.0;
        for (int i = 0; i < N; ++i) {
            lo = fminf(lo, a[i]);
            hi = fmaxf(hi, a[i]);
            s = __dadd_rn(s, (double)a[i]);
        }
        kept[p] = N; nlo[p] = rlo; nhi[p] = rhi;
        pmin[p] = lo; pmax[p] = hi;
        kmean[p] = (float)(s / N);
    }
}

// Siril mean_and_reject: weighted mean (double sums) of the values inside [pmin, pmax], weights =
// drizzle * feather * image weight; when the weights vanish, the sum of the positive in-range
// values / kept. plain = 1 (no drizzle, feather or weighting): mean of the kept values.
extern "C" __global__ void wmean(const float* __restrict__ V, const float* __restrict__ W,
                                 const double* __restrict__ imgw, const int K, const long long P,
                                 const float* __restrict__ pmin, const float* __restrict__ pmax,
                                 const int* __restrict__ kept, const float* __restrict__ kmean,
                                 const int plain, float* __restrict__ out) {
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int nk = kept[p];
    if (nk == 0) { out[p] = 0.f; return; }
    if (plain) { out[p] = kmean[p]; return; }
    const float lo = pmin[p], hi = pmax[p];
    double s = 0.0, nrm = 0.0, su = 0.0;
    for (int k = 0; k < K; ++k) {
        const float v = V[(long long)k * P + p];
        if (v >= lo && v <= hi) {           // false for NaN
            const double w = __dmul_rn((double)W[(long long)k * P + p], imgw[k]);
            s = __dadd_rn(s, __dmul_rn((double)v, w));
            nrm = __dadd_rn(nrm, w);
            if (v > 0.f) su = __dadd_rn(su, (double)v);
        }
    }
    out[p] = (nrm != 0.0 && s != 0.0) ? (float)(s / nrm) : (float)(su / nk);
}

// Dense (K, R*C) tile from per-frame strip blocks stored back to back in one flat buffer.
extern "C" __global__ void gather_tile(const float* __restrict__ vals, const float* __restrict__ wts,
                                       const long long* __restrict__ foff, const int* __restrict__ fnr,
                                       const int* __restrict__ fwid, const int* __restrict__ fx0,
                                       const int* __restrict__ fcy, const int K, const int R, const int C,
                                       const int r0, const int c0,
                                       float* __restrict__ V, float* __restrict__ W) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long P = (long long)R * C;
    if (i >= (long long)K * P) return;
    const int k = (int)(i / P);
    const int pix = (int)(i - (long long)k * P);
    const int ly = r0 + pix / C - fcy[k], lx = c0 + pix % C - fx0[k];
    float v = __int_as_float(0x7fc00000), w = 0.f;
    if (ly >= 0 && ly < fnr[k] && lx >= 0 && lx < fwid[k]) {
        const long long o = foff[k] + (long long)ly * fwid[k] + lx;
        v = vals[o]; w = wts[o];
    }
    V[i] = v; W[i] = w;
}

// ---- Siril / CFITSIO FnNoise1_float (algos/quantize.c), one thread per image row ----
// differences between consecutive non-null pixels; mean/sigma over the non-zero differences
// (FnMeanSigma_float, double sums, population sigma); up to 3 passes of 5-sigma clipping.
__device__ void diff_moments(const float* __restrict__ row, const int nx, const int ncrit,
                             const float* cm, const double* cs, long long &nv, long long &ng,
                             double &sum, double &sum2) {
    nv = 0; ng = 0; sum = 0.0; sum2 = 0.0;
    int ii = 0;
    while (ii < nx && (row[ii] == 0.f || isnan(row[ii]))) ii++;
    if (ii == nx) return;
    float v1 = row[ii];
    for (ii++; ii < nx; ii++) {
        const float v = row[ii];
        if (v == 0.f || isnan(v)) continue;
        const float d = __fsub_rn(v1, v);
        v1 = v;
        bool keep = true;
        for (int t = 0; t < ncrit; t++) {
            if (!((double)fabsf(__fsub_rn(d, cm[t])) < 5.0 * cs[t])) { keep = false; break; }
        }
        if (!keep) continue;
        nv++;
        if (d != 0.f) {
            ng++;
            const double x = (double)d;
            sum = __dadd_rn(sum, x);
            sum2 = __dadd_rn(sum2, __dmul_rn(x, x));
        }
    }
}

__device__ void mean_sigma(long long ng, double sum, double sum2, double &mean, double &sd) {
    if (ng > 1) {                           // no FMA contraction: match the CPU's rounding
        const double n = (double)ng;
        mean = __ddiv_rn(sum, n);
        sd = sqrt(__dsub_rn(__ddiv_rn(sum2, n), __dmul_rn(mean, mean)));
    }
    else if (ng == 1) { mean = sum; sd = 0.0; }
    else { mean = 0.0; sd = 0.0; }
}

extern "C" __global__ void noise1_rows(const float* __restrict__ img, const int nx, const int ny,
                                       double* __restrict__ row_sd, int* __restrict__ row_ok) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    row_ok[j] = 0;
    if (nx < 3) return;
    const float* row = img + (long long)j * nx;
    float cm[4]; double cs[4];
    long long nv, ng; double sum, sum2, mean, sd;
    diff_moments(row, nx, 0, cm, cs, nv, ng, sum, sum2);
    if (nv < 2) return;
    mean_sigma(ng, sum, sum2, mean, sd);
    if (sd > 0.0) {
        for (int iter = 0; iter < 3; iter++) {
            cm[iter] = (float)mean; cs[iter] = sd;
            long long kk, ng2;
            diff_moments(row, nx, iter + 1, cm, cs, kk, ng2, sum, sum2);
            if (kk == nv) break;
            nv = kk;
            mean_sigma(ng2, sum, sum2, mean, sd);
        }
    }
    row_sd[j] = sd;
    row_ok[j] = 1;
}

// ---- feather-mask downscale: OpenCV INTER_LINEAR on 8-bit, fixed point (11-bit coefficients)
extern "C" __global__ void resize_u8_linear(const unsigned char* __restrict__ src, const int sw,
                                            const int sh, const int dw, const int dh,
                                            const int* __restrict__ xofs, const short* __restrict__ ax,
                                            const int* __restrict__ yofs, const short* __restrict__ by,
                                            unsigned char* __restrict__ dst) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)dw * dh) return;
    const int y = (int)(i / dw), x = (int)(i % dw);
    const int sx = xofs[x], sy = yofs[y];
    const int sx1 = min(sx + 1, sw - 1), sy1 = min(sy + 1, sh - 1);
    const int a0 = ax[2 * x], a1 = ax[2 * x + 1], b0 = by[2 * y], b1 = by[2 * y + 1];
    const unsigned char* r0 = src + (long long)sy * sw;
    const unsigned char* r1 = src + (long long)sy1 * sw;
    const int h0 = r0[sx] * a0 + r0[sx1] * a1;
    const int h1 = r1[sx] * a0 + r1[sx1] * a1;
    const long long v = (long long)h0 * b0 + (long long)h1 * b1 + (1LL << 21);
    int o = (int)(v >> 22);
    dst[i] = (unsigned char)(o < 0 ? 0 : (o > 255 ? 255 : o));
}
"""

_mod = cp.RawModule(code=_SOURCE)
reject = _mod.get_function("reject")
reject_w = _mod.get_function("reject_w")
reject_fixup = _mod.get_function("reject_fixup")
wmean = _mod.get_function("wmean")
gather_tile = _mod.get_function("gather_tile")
noise1_rows = _mod.get_function("noise1_rows")
resize_u8_linear = _mod.get_function("resize_u8_linear")


def launch(fn, n: int, *args, block: int = 256) -> None:
    fn(((n + block - 1) // block,), (block,), args)


REJECT_SHARED = 48 * 1024
REJECT_WARP_MIN = 96            # below ~100 values a column, a thread per pixel is as fast or faster
FIXUP_WORK = 1 << 20            # floats of scratch for reject_fixup (its thread count follows from k)


def reject_columns(S, V, k: int, p: int, mode: int, siglow: float, sighigh: float,
                   pmin, pmax, kept, nvalid, nlo, nhi, kmean, warp: bool | None = None) -> None:
    """Rejection over the sorted (k, p) columns of S, by reject_w (a warp per pixel, its column in shared
    memory) or reject (a thread per pixel); both give the same bits. V: the same values in frame order (S =
    sort_columns(V)). Where Siril's keep-4 guard cuts a pass short at 9+ values, the outliers it drops depend
    on that order, so the kernels list those (rare) pixels and reject_fixup redoes them from V, without a
    round trip to the host. warp=None picks the faster kernel for k (always reject with
    GPUSTACKER_THREAD_REJECT=1); True or False forces one where possible."""
    outs = (pmin, pmax, kept, nvalid, nlo, nhi, kmean)
    nbind, bind = cp.zeros(1, cp.int32), cp.empty(p, cp.int32)
    args = (np.int32(mode), np.float32(siglow), np.float32(sighigh)) + outs + (nbind, bind)
    ppb = 8                                     # pixels (warps) per block
    while ppb > 1 and ppb * (k + 32) * 4 > REJECT_SHARED:
        ppb //= 2
    # column stride: the block's loads (ppb pixels x 32 // ppb rows per warp) then hit distinct banks
    kpad = k + (32 // ppb - k) % 32 if ppb > 1 else k
    if warp is None:
        warp = k >= REJECT_WARP_MIN and os.environ.get("GPUSTACKER_THREAD_REJECT") != "1"
    if not warp or ppb * kpad * 4 > REJECT_SHARED:
        launch(reject, p, S, np.int32(k), np.int64(p), *args)
    else:
        reject_w(((p + ppb - 1) // ppb,), (32 * ppb,), (S, np.int32(k), np.int64(p), np.int32(kpad)) + args,
                 shared_mem=ppb * kpad * 4)
    blocks = max(1, min(64, FIXUP_WORK // (64 * k)))            # 64 to 4096 threads, each with k floats
    work = cp.empty(blocks * 64 * k, cp.float32)
    V = cp.ascontiguousarray(V, dtype=cp.float32)
    reject_fixup((blocks,), (64,), (V, np.int32(k), np.int64(p), np.int32(mode), np.float32(siglow),
                                   np.float32(sighigh), nbind, bind, work) + outs)
