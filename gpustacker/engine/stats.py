# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
# Contains code ported from Siril, RawTherapee and CFITSIO; see THIRD-PARTY-NOTICES.txt
"""Per-frame statistics as Siril computes them, plus an on-disk cache.

  location, scale   IKSSlite (Siril 1.4 STATS_NORM) on the non-null pixels of each layer, with Siril's
                    histogram medians: equal to Siril's own values at full precision
  bgnoise           CFITSIO FnNoise1 as ported in Siril algos/quantize.c (used by -weight=noise)
  ngood             non-zero, non-NaN pixel count

The cache is plain JSON so the UI can read it without the GPU stack.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

CACHE_VERSION = 2                                  # 2: location/scale bit-exact with Siril


@dataclass
class FrameStats:
    loc: list = field(default_factory=list)       # per layer, None = unknown
    scale: list = field(default_factory=list)
    noise: list = field(default_factory=list)
    ngood: list = field(default_factory=list)
    bad: str = ""                                  # non-empty: why this frame can't be stacked

    def has_norm(self, layers: int) -> bool:
        return len(self.scale) >= layers and all(self.scale[c] is not None and self.scale[c] > 0
                                                 and self.loc[c] is not None for c in range(layers))

    def has_noise(self, layers: int) -> bool:
        return len(self.noise) >= layers and all(n is not None and n > 0 for n in self.noise[:layers])


def file_signature(path: str) -> list[int]:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


class StatsCache:
    """JSON cache of FrameStats keyed by file number, invalidated by file size and mtime."""

    def __init__(self, path: str):
        self.path = path
        self.entries: dict[str, dict] = {}
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("version") == CACHE_VERSION:
                self.entries = d.get("frames", {})
        except (OSError, ValueError):
            pass

    def get(self, filenum: int, sig: list[int]) -> FrameStats | None:
        e = self.entries.get(str(filenum))
        if not e or e.get("sig") != sig:
            return None
        return FrameStats(loc=e.get("loc", []), scale=e.get("scale", []), noise=e.get("noise", []),
                          ngood=e.get("ngood", []), bad=e.get("bad", ""))

    def put(self, filenum: int, sig: list[int], st: FrameStats) -> None:
        self.entries[str(filenum)] = {"sig": sig, **asdict(st)}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "frames": self.entries}, f, separators=(",", ":"))
        os.replace(tmp, self.path)


# ----------------------------------------------------------------------------- GPU estimators
# Siril 1.4.4 for -norm=addscale (statistics_internal_float with STATS_NORM, algos/statistics_float.c): median
# and MAD of the non-null pixels by histogram_median_float, then IKSSlite. Ported operation by operation, in
# float where Siril works in float, with _rn intrinsics so that nvcc cannot fuse anything into FMAs (Siril's
# Windows build targets plain x86-64: no FMA, no fast-math), and its double sums added in Siril's order.

_SRC = r"""
// ---- RawTherapee's findMinMaxPercentile at 50 % (Siril's histogram_median_float), on the device.
// Order-preserving uint32 keys of floats (no NaNs, no -0 here) for min/max with integer atomics.
__device__ __forceinline__ unsigned int fkey(const float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float unkey(const unsigned int k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// grid-stride loop
#define GRID_LOOP(i, n) for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); \
                             i += (long long)gridDim.x * blockDim.x)

// mm = {key of min, key of max} of x, or of |x - *center| (written to dev: siril_stats_float_mad's deviations)
extern "C" __global__ void hm_minmax(const float* __restrict__ x, const long long n, const float* __restrict__ center,
                                     float* __restrict__ dev, unsigned int* __restrict__ mm) {
    unsigned int lo = 0xffffffffu, hi = 0u;
    const float c = center ? *center : 0.f;
    GRID_LOOP(i, n) {
        float v = x[i];
        if (center) { v = fabsf(__fsub_rn(v, c)); dev[i] = v; }
        const unsigned int k = fkey(v);
        lo = min(lo, k);
        hi = max(hi, k);
    }
    for (int o = 16; o > 0; o >>= 1) {
        lo = min(lo, __shfl_down_sync(0xffffffffu, lo, o));
        hi = max(hi, __shfl_down_sync(0xffffffffu, hi, o));
    }
    if ((threadIdx.x & 31) == 0) { atomicMin(&mm[0], lo); atomicMax(&mm[1], hi); }
}

// min(65536, n) bins over [min, max], bin = static_cast<uint16_t>(scale * (v - minVal)); false if max == min
__device__ __forceinline__ bool hm_scale(const unsigned int* __restrict__ mm, const float nbins_m1, float* lo,
                                         float* scale) {
    *lo = unkey(mm[0]);
    const float d = __fsub_rn(unkey(mm[1]), *lo);
    *scale = __fdiv_rn(nbins_m1, d);
    return fabsf(d) != 0.f;
}
__device__ __forceinline__ unsigned int hm_bin(const float v, const float lo, const float scale) {
    const float f = __fmul_rn(scale, __fsub_rn(v, lo));    // x86 cvttss2si: out of range or NaN -> INT_MIN
    return (unsigned short)((f >= -2147483648.f && f < 2147483648.f) ? (int)f : (int)0x80000000);
}
__device__ __forceinline__ void warp_inc(unsigned int* sh, const unsigned int b) {
    const unsigned int peers = __match_any_sync(__activemask(), b);
    if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&sh[b], __popc(peers));
}

// The histogram in two levels (blocks of 256 threads; the data is too concentrated for 65536 global counters):
// counts of the 256 coarse bins (bin >> 8)...
extern "C" __global__ void hm_coarse(const float* __restrict__ x, const long long n,
                                     const unsigned int* __restrict__ mm, const float nbins_m1,
                                     unsigned int* __restrict__ hc) {
    __shared__ unsigned int sh[256];
    sh[threadIdx.x] = 0;
    __syncthreads();
    float lo, scale;
    if (hm_scale(mm, nbins_m1, &lo, &scale))
        GRID_LOOP(i, n)
            warp_inc(sh, hm_bin(x[i], lo, scale) >> 8);
    __syncthreads();
    if (sh[threadIdx.x]) atomicAdd(&hc[threadIdx.x], sh[threadIdx.x]);
}

// ...then the 256 bins of the coarse bin where the running count reaches n/2 (as float, like the C loop), and
// cinfo = {that coarse bin, count below it}
extern "C" __global__ void hm_fine(const float* __restrict__ x, const long long n, const unsigned int* __restrict__ mm,
                                   const float nbins_m1, const unsigned int* __restrict__ hc,
                                   unsigned int* __restrict__ hf, unsigned long long* __restrict__ cinfo) {
    __shared__ unsigned int sh[256];
    __shared__ unsigned long long sc[256];
    __shared__ unsigned int csel;
    const int t = threadIdx.x;
    float lo, scale;
    if (!hm_scale(mm, nbins_m1, &lo, &scale)) return;
    const float thresh = __fmul_rn(0.5f, __ull2float_rn((unsigned long long)n));
    sh[t] = 0;
    sc[t] = hc[t];
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        const unsigned long long v = t >= off ? sc[t - off] : 0ULL;
        __syncthreads();
        sc[t] += v;
        __syncthreads();
    }
    if (__ull2float_rn(sc[t]) >= thresh && (t == 0 || __ull2float_rn(sc[t - 1]) < thresh)) {
        csel = t;
        if (blockIdx.x == 0) { cinfo[0] = t; cinfo[1] = sc[t] - hc[t]; }
    }
    __syncthreads();
    const unsigned int c = csel;
    GRID_LOOP(i, n) {
        const unsigned int b = hm_bin(x[i], lo, scale);
        if ((b >> 8) == c) warp_inc(sh, b & 255);
    }
    __syncthreads();
    if (sh[t]) atomicAdd(&hf[t], sh[t]);
}

// findMinMaxPercentile's search and interpolation, all in float: walk the bins while the running count, as a
// float, is below n/2; place the median inside the last bin linearly; back to data units; clamp to [min, max]
extern "C" __global__ void hm_median(const unsigned int* __restrict__ hf, const unsigned long long* __restrict__ cinfo,
                                     const long long n, const unsigned int* __restrict__ mm, const float nbins_m1,
                                     float* __restrict__ med) {
    float lo, scale;
    const float hi = unkey(mm[1]);
    if (!hm_scale(mm, nbins_m1, &lo, &scale)) { *med = lo; return; }
    const float thresh = __fmul_rn(0.5f, __ull2float_rn((unsigned long long)n));
    const unsigned int c0bin = (unsigned int)cinfo[0] * 256;
    unsigned long long count = cinfo[1];
    unsigned int k = c0bin;
    while (__ull2float_rn(count) < thresh) count += hf[(k++) - c0bin];
    const float c0 = __fsub_rn(__ull2float_rn(count), thresh);
    const float c1 = __fsub_rn(thresh, __ull2float_rn(count - hf[k - 1 - c0bin]));
    float m = __fdiv_rn(__fadd_rn(__fmul_rn(c1, (float)k), __fmul_rn(c0, (float)(k - 1))), __fadd_rn(c0, c1));
    m = __fadd_rn(__fdiv_rn(m, scale), lo);
    *med = fmaxf(lo, fminf(m, hi));
}

// ---- the float terms of Siril's siril_stats_float_bwmv (rows 0 and 1 of out); their sums are added in double
extern "C" __global__ void bwmv_terms(const float* __restrict__ x, const long long n, const float med,
                                      const float factor, float* __restrict__ out) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float i_med = __fsub_rn(x[i], med);
    const float yi = __fmul_rn(i_med, factor);
    const float yi2 = fabsf(yi) < 1.f ? __fmul_rn(yi, yi) : 1.f;
    const float w = __fsub_rn(1.f, yi2);
    const float a = __fmul_rn(i_med, __fmul_rn(w, w));
    out[i] = __fmul_rn(a, a);
    out[n + i] = __fmul_rn(w, __fsub_rn(1.f, __fmul_rn(5.f, yi2)));
}

// ---- t[0] + t[1] + ... added one by one in double, exactly, in parallel (see sequential_sums).
#define NOKEY 0x7fffffff
// run key of a predicted partial sum: binade and sign; zero, subnormal and non-finite sums get NOKEY
__device__ __forceinline__ int run_key(const double q, int* e) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(q);
    const int fe = (int)((b >> 52) & 0x7ff);
    *e = fe - 1022;                                  // frexp's exponent: |q| in [2^(e-1), 2^e), grid 2^(e-53)
    if (fe == 0 || fe == 0x7ff) return NOKEY;
    return (b >> 63) ? -*e - 4096 : *e;
}

// t / 2^(e-53) rounded to the nearest integer, from the float's bits; exactly halfway: floor, tie = true.
// false when the result could reach 2^60 or t is not finite.
__device__ __forceinline__ bool scaled_round(const float tf, const int e, long long* r, bool* tie) {
    const unsigned int u = __float_as_uint(tf);
    const int fe = (u >> 23) & 0xff;
    unsigned long long M = u & 0x7fffffu;
    *tie = false;
    if (fe == 255) return false;
    if (fe == 0 && M == 0) { *r = 0; return true; }
    int ex = fe;
    if (fe == 0) ex = 1; else M |= 0x800000u;
    const int sh = ex - 150 + 53 - e;                // |t| = M 2^(ex-150), so |t| / grid = M 2^sh
    const bool neg = u >> 31;
    long long v;
    if (sh >= 0) {
        if (sh > 35) return false;
        v = (long long)(M << sh);
    } else if (-sh >= 25) {
        v = 0;                                       // M < 2^24: under one half
    } else {
        const int k = -sh;
        const unsigned long long ip = M >> k, rem = M & ((1ULL << k) - 1), half = 1ULL << (k - 1);
        if (rem == half) { *tie = true; *r = neg ? -(long long)ip - 1 : (long long)ip; return true; }
        v = (long long)(ip + (rem > half ? 1ULL : 0ULL));
    }
    *r = neg ? -v : v;
    return true;
}

__device__ __forceinline__ long long upper(const long long* __restrict__ a, long long n, long long v) {
    long long lo = 0, hi = n;                        // number of a[] <= v (a sorted)
    while (lo < hi) { const long long m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
    return lo;
}

// One thread per term after the warm-up columns (flat index i over rows x (n - w)). Q = float64 prefix sums
// of each row, in any order: they only predict the binade of the running sum. A run starts at a row's first
// term here and wherever the predicted binade or sign changes; its first step is added on the CPU. Every
// other step adds r = t / grid rounded to an integer (ties: floor, fixed by seq_ties).
extern "C" __global__ void seq_prep(const float* __restrict__ T, const double* __restrict__ Q, const long long n,
                                    const long long w, const long long N, long long* __restrict__ r,
                                    long long* __restrict__ starts, long long* __restrict__ ties,
                                    unsigned long long* __restrict__ counts) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    const long long nn = n - w, j = (i / nn) * n + w + i % nn;
    int e, ep;
    const int key = run_key(Q[j], &e);
    const bool st = (i % nn) == 0 || key == NOKEY || run_key(Q[j - 1], &ep) != key;
    if (key == NOKEY && !isfinite(Q[j])) atomicAdd(&counts[2], 1ULL);
    if (st) {
        r[i] = 0;
        if (!isfinite(T[j])) atomicAdd(&counts[2], 1ULL);
        starts[atomicAdd(&counts[0], 1ULL)] = i;
        return;
    }
    long long ri;
    bool tie;
    if (!scaled_round(T[j], e, &ri, &tie)) atomicAdd(&counts[2], 1ULL);
    r[i] = ri;
    if (tie) ties[atomicAdd(&counts[1], 1ULL)] = i;
}

// Round-half-even of a tie needs the parity of the partial sum before it. A tie leaves the sum even, so that
// parity is the one of the r's since the previous tie of the run (no tie in between); for a run's first tie
// it is relative to the run start, whose own parity the CPU adds (adj). d = 1 where the tie rounds up.
extern "C" __global__ void seq_ties(const long long* __restrict__ ti, const long long nt,
                                    const long long* __restrict__ s0, const long long K,
                                    const unsigned long long* __restrict__ R, const long long* __restrict__ r,
                                    long long* __restrict__ d) {
    const long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (j >= nt) return;
    const long long i = ti[j];
    long long ref = s0[upper(s0, K, i) - 1];
    if (j > 0 && ti[j - 1] > ref) ref = ti[j - 1];
    d[j] = ((long long)(R[i - 1] - R[ref]) + r[i]) & 1;
}

// Per run, for the CPU: total of its later steps (ties rounded as after an even start), first tie, change of
// that tie after an odd start, number of later steps, grid exponent, first term, number of ties, first index.
extern "C" __global__ void seq_runs(const long long* __restrict__ s0, const long long K, const long long N,
                                    const unsigned long long* __restrict__ R, const long long* __restrict__ ti,
                                    const long long nt, const long long* __restrict__ d,
                                    const long long* __restrict__ dc, const double* __restrict__ Q,
                                    const float* __restrict__ T, const long long n, const long long w,
                                    long long* __restrict__ info) {
    const long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k >= K) return;
    const long long st = s0[k], en = (k + 1 < K) ? s0[k + 1] - 1 : N - 1;
    const long long c0 = upper(ti, nt, st), c1 = upper(ti, nt, en);   // ties of the run: ti[c0 .. c1-1]
    long long D = (long long)(R[en] - R[st]), ft = -1, adj = 0;
    if (c1 > c0) {
        D += dc[c1 - 1] - (c0 ? dc[c0 - 1] : 0);
        ft = ti[c0];
        adj = 1 - 2 * d[c0];
    }
    const long long j = (st / (n - w)) * n + w + st % (n - w);
    int e;
    run_key(Q[j], &e);
    info[k] = D;
    info[K + k] = ft;
    info[2 * K + k] = adj;
    info[3 * K + k] = en - st;
    info[4 * K + k] = e;
    info[5 * K + k] = __double_as_longlong((double)T[j]);
    info[6 * K + k] = c1 - c0;
    info[7 * K + k] = st;
}

// Each later step's partial sum, in units of its run's grid, must lie strictly inside the run's binade: then
// it is exactly the IEEE double sum (induction from the run's exact first step), and a run's last partial sum
// must be the one the CPU carried on with. Ties move a partial sum by at most their count plus one, so only
// partial sums near the binade's edges (and run ends) need them counted.
extern "C" __global__ void seq_check(const unsigned long long* __restrict__ R, const long long N,
                                     const long long* __restrict__ s0, const long long K,
                                     const long long* __restrict__ A, const long long* __restrict__ info,
                                     const long long* __restrict__ ti, const long long nt,
                                     const long long* __restrict__ dc, int* __restrict__ bad) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    const long long s = upper(s0, K, i) - 1;
    const long long st = s0[s];
    if (st == i) return;
    const bool last = (s + 1 < K) ? (i == s0[s + 1] - 1) : (i == N - 1);
    const long long P = 1LL << 52, P2 = 1LL << 53;
    long long a = A[2 * s] + (long long)(R[i] - R[st]);
    const long long lo = a - 1, hi = a + info[6 * K + s] + 1;
    if (!last && ((lo > P && hi < P2) || (lo > -P2 && hi < -P))) return;
    const long long c1 = upper(ti, nt, i), c0 = upper(ti, nt, st);
    a += (c1 ? dc[c1 - 1] : 0) - (c0 ? dc[c0 - 1] : 0);
    const long long ft = info[K + s];
    if (ft >= 0 && i >= ft && (A[2 * s] & 1)) a += info[2 * K + s];
    const long long m = a < 0 ? -a : a;
    if (!(m > P && m < P2) || (last && a != A[2 * s + 1])) atomicAdd(bad, 1);
}
"""
_mod = None
_SEQ_WARMUP = 4096                                    # first terms of each row: summed on the CPU


def _fn(name):
    import cupy as cp
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC)
    return _mod.get_function(name)


def _launch(name, n, *args, block=256, max_grid=0):
    if n > 0:
        grid = (n + block - 1) // block
        _fn(name)((min(grid, max_grid) if max_grid else grid,), (block,), args)


def _f32(x):
    import cupy as cp
    return cp.ascontiguousarray(x, dtype=cp.float32)


_SLOT = 520         # uint32 per histogram median: min/max keys, median, coarse bin + count below, 256 + 256 counts


def _workspace(slots: int):
    """Device scratch for `slots` histogram medians; the medians (float32) are column 2."""
    import cupy as cp
    ws = cp.zeros((slots, _SLOT), cp.uint32)
    ws[:, 0] = 0xFFFFFFFF
    return ws


def _medians(ws) -> list[float]:
    import numpy as np
    return ws[:, 2].get().view(np.float32).tolist()


def _hmed(x, ws, slot, center=None):
    """histogram_median_float of x, or of |x - *center| (siril_stats_float_mad), into ws's slot, on the device;
    -> the median (a 1-element CuPy float32 view)."""
    import cupy as cp
    import numpy as np
    w = ws[slot]
    mm, med, cinfo, hc, hf = w[0:2], w[2:3].view(cp.float32), w[4:8].view(cp.uint64), w[8:264], w[264:520]
    n = x.size
    nbins_m1 = np.float32(min(65536, n) - 1)
    null = np.uint64(0)
    dev = cp.empty_like(x) if center is not None else None
    _launch("hm_minmax", n, x, np.int64(n), null if dev is None else center, null if dev is None else dev, mm,
            max_grid=1024)
    data = x if dev is None else dev
    _launch("hm_coarse", n, data, np.int64(n), mm, nbins_m1, hc, max_grid=1024)
    _launch("hm_fine", n, data, np.int64(n), mm, nbins_m1, hc, hf, cinfo, max_grid=1024)
    _fn("hm_median")((1,), (1,), (hf, cinfo, np.int64(n), mm, nbins_m1, med))
    return med


def histogram_median(x) -> float:
    """Siril's histogram_median_float of 1-D CuPy float32 data without NaNs: RawTherapee's
    findMinMaxPercentile at 50 %, i.e. a histogram of min(65536, n) bins over [min, max] and the median
    interpolated linearly inside its bin, all in float. Within a bin width of the exact median, not equal."""
    ws = _workspace(1)
    _hmed(_f32(x), ws, 0)
    return _medians(ws)[0]


def mad(x, m: float) -> float:
    """siril_stats_float_mad: histogram_median_float of |x - (float)m| (float deviations)."""
    import cupy as cp
    ws = _workspace(1)
    _hmed(_f32(x), ws, 0, cp.asarray([m], cp.float32))
    return _medians(ws)[0]


def sequential_sums(T) -> list[float]:
    """Row sums of a (m, n) CuPy float32 array, each added left to right in double: bit-identical to the C loop
    `double s = 0; for (i = 0; i < n; i++) s += t[i];` (Siril's single-threaded sums), but in parallel.

    Inside a run of steps whose partial sums stay in one binade [2^e, 2^(e+1)), every partial sum is a multiple
    of the grid g = 2^(e-52), so each step adds t/g rounded to an integer: independent of the running sum except
    on ties (exactly halfway), which round to the even neighbour. A tie leaves the sum even, so the parity before
    a tie is that of the steps since the previous tie, and only a run's first tie depends on how the run starts.
    Runs are predicted from float64 prefix sums; the first step of every run (a binade change) is added on the
    CPU; every other partial sum is then checked to lie strictly inside its run's binade, which makes it equal
    to IEEE's. The first terms of each row (where the binade changes often), and anything unexpected, are
    summed on the CPU."""
    import math

    import cupy as cp
    import numpy as np

    T = _f32(T)
    m, n = T.shape
    w = min(n, _SEQ_WARMUP)
    if n == w:
        return _cpu_sums(T)
    N = m * (n - w)
    Q = cp.empty((m, n), cp.float64)
    for k in range(m):
        cp.cumsum(T[k], dtype=cp.float64, out=Q[k])
    r = cp.empty(N, cp.int64)
    starts = cp.empty(N, cp.int64)
    ties = cp.empty(N, cp.int64)
    counts = cp.zeros(3, cp.uint64)
    _launch("seq_prep", N, T, Q, np.int64(n), np.int64(w), np.int64(N), r, starts, ties, counts)
    R = cp.cumsum(r.view(cp.uint64))                 # modular: only differences inside a run are used
    K, nt, bad = (int(v) for v in counts.get())
    if bad:
        return _cpu_sums(T)
    s0 = cp.sort(starts[:K])
    ti = cp.sort(ties[:nt])
    d = cp.empty(nt, cp.int64)
    _launch("seq_ties", nt, ti, np.int64(nt), s0, np.int64(K), R, r, d)
    dc = cp.cumsum(d) if nt else cp.zeros(1, cp.int64)
    info = cp.empty(8 * K, cp.int64)
    _launch("seq_runs", K, s0, np.int64(K), np.int64(N), R, ti, np.int64(nt), d, dc, Q, T, np.int64(n),
            np.int64(w), info)
    del Q
    inf = info.get().reshape(8, K)
    D0, _, adj, steps, ex = (row.tolist() for row in inf[:5])
    tstart = inf[5].view(np.float64).tolist()
    out = _cpu_sums(T[:, :w])
    A = [0] * (2 * K)                                # per run: first and last partial sum, in grid units
    S = 0.0
    for k, i0 in enumerate(inf[7].tolist()):
        row = i0 // (n - w)
        if i0 % (n - w) == 0:
            S = out[row]
        S = S + tstart[k]                            # the run's first step: exact IEEE double addition
        if steps[k] > 0:
            g = math.ldexp(1.0, ex[k] - 53)
            a = S / g
            if not a.is_integer() or not abs(a) < 2.0 ** 53:
                return _cpu_sums(T)
            A[2 * k] = int(a)
            A[2 * k + 1] = int(a) + D0[k] + (adj[k] if int(a) & 1 else 0)
            S = float(A[2 * k + 1]) * g
        out[row] = S
    badc = cp.zeros(1, cp.int32)
    _launch("seq_check", N, R, np.int64(N), s0, np.int64(K), cp.asarray(np.array(A, np.int64)), info, ti,
            np.int64(nt), dc, badc)
    return _cpu_sums(T) if int(badc.get()[0]) else out


def _cpu_sums(T) -> list[float]:
    import numpy as np
    return np.cumsum(T.get(), axis=1, dtype=np.float64)[:, -1].tolist() if T.shape[1] else [0.0] * T.shape[0]


def bwmv(x, mad_: float, median: float) -> float:
    """siril_stats_float_bwmv: biweight midvariance of x around median, terms in float, sums in double."""
    import cupy as cp
    import numpy as np
    if not mad_ > 0:
        return 0.0
    x = _f32(x)
    n = x.size
    terms = cp.empty((2, n), cp.float32)
    _launch("bwmv_terms", n, x, np.int64(n), np.float32(median), np.float32(1) / (np.float32(9) * np.float32(mad_)),
            terms)
    up, down = sequential_sums(terms)
    return n * (up / (down * down)) if down else 0.0


def ikss_lite(x, median: float, mad_: float) -> tuple[float, float] | None:
    """Siril's IKSSlite on 1-D non-null samples x (CuPy float32, Siril's pixel order for a bit-exact scale)
    with the median and MAD Siril passes in: (location, scale), None where Siril fails (MAD 0)."""
    import math

    import numpy as np
    x = _f32(x)
    m, d = float(np.float32(median)), float(np.float32(mad_))
    k = x[(x >= np.float32(m - 6.0 * d)) & (x <= np.float32(m + 6.0 * d))]   # bounds: double, stored as float
    if k.size == 0:
        return None
    ws = _workspace(2)
    _hmed(k, ws, 1, _hmed(k, ws, 0))
    loc, s = _medians(ws)
    if s == 0.0:
        return None
    return loc, math.sqrt(bwmv(k, s, loc)) * .991


def norm_stats(x) -> tuple[float, float] | None:
    """(location, scale) of the non-null samples x (CuPy float32, NaN-free) exactly as Siril 1.4.4 computes
    them for -norm=addscale; x in Siril's memory order (FITS row order) for a bit-exact scale."""
    x = _f32(x)
    if x.size < 2:
        return None
    ws = _workspace(2)
    _hmed(x, ws, 1, _hmed(x, ws, 0))
    return ikss_lite(x, *_medians(ws))


def bgnoise(plane) -> float:
    """Siril's bgnoise for one layer (CuPy (h, w) float32, any row order)."""
    import cupy as cp
    import numpy as np

    from .kernels import launch, noise1_rows

    ny, nx = plane.shape
    sd = cp.zeros(ny, cp.float64)
    ok = cp.zeros(ny, cp.int32)
    launch(noise1_rows, ny, cp.ascontiguousarray(plane), np.int32(nx), np.int32(ny), sd, ok, block=64)
    v = cp.sort(sd[ok == 1]).get()
    n = v.size
    if n == 0:
        x = 0.0
    elif n == 1:
        x = float(v[0])
    else:
        x = (float(v[(n - 1) // 2]) + float(v[n // 2])) / 2.0
    return 0.70710678 * x


def measure_planes(planes, need_norm: bool, need_noise: bool) -> FrameStats:
    """FrameStats for a (C, h, w) CuPy float32 image (rows in FITS order, as Siril holds them in memory)."""
    import cupy as cp

    st = FrameStats()
    for c in range(planes.shape[0]):
        x = planes[c]
        valid = (x != 0) & ~cp.isnan(x)
        ngood = int(cp.count_nonzero(valid))
        st.ngood.append(ngood)
        loc = scale = None
        if need_norm and ngood > 1:
            r = norm_stats(x[valid])
            if r is not None:
                loc, scale = r
        st.loc.append(loc)
        st.scale.append(scale)
        st.noise.append(bgnoise(x) if need_noise and ngood > 2 else None)
    return st
