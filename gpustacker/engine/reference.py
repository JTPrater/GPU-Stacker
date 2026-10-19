# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
# Contains code ported from Siril; see THIRD-PARTY-NOTICES.txt
"""Literal CPU port of Siril 1.4.4's per-pixel rejection and mean (slow; for tests only).

Follows stacking/rejection_float.c (apply_rejection_float), stacking/median_and_mean.c
(mean_and_reject) and algos/sorting.c (quickmedian_float, sortnet_median_float), including
their in-place reordering of the stack, in float32 arithmetic where Siril uses float.
"""
from __future__ import annotations

import numpy as np

f32 = np.float32

_NETS = {
    2: [(0, 1)],
    3: [(0, 1), (1, 2), (0, 1)],
    4: [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)],
    5: [(0, 1), (2, 3), (1, 3), (2, 4), (0, 2), (1, 4), (1, 2), (3, 4), (2, 3)],
    6: [(0, 1), (2, 3), (4, 5), (0, 2), (3, 5), (1, 4), (0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (2, 3)],
    7: [(1, 2), (3, 4), (5, 6), (0, 2), (4, 6), (3, 5), (2, 6), (1, 5), (0, 4), (2, 5), (0, 3), (2, 4),
        (1, 3), (0, 1), (2, 3), (4, 5)],
    8: [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2), (5, 6), (0, 4), (1, 5),
        (2, 6), (3, 7), (2, 4), (3, 5), (1, 2), (3, 4), (5, 6)],
}


def quickmedian(a: list, n: int) -> float:
    """quickmedian_float on a[:n], reordering a in place like Siril."""
    if n == 0:
        return 0.0
    if n < 9:
        if n == 1:
            return float(a[0])
        for i, j in _NETS[n]:
            if a[i] > a[j]:
                a[i], a[j] = a[j], a[i]
        k = n // 2
        return float(f32(a[k - 1] + a[k])) / 2.0 if n % 2 == 0 else float(a[k])
    k = n // 2
    left, right = 0, n - 1
    while left < right:
        pindex = (left + right) // 2
        pivot = a[pindex]
        a[pindex] = a[right]
        a[right] = pivot
        pindex = left
        for i in range(left, right):
            if a[i] < pivot:
                a[pindex], a[i] = a[i], a[pindex]
                pindex += 1
        a[right] = a[pindex]
        a[pindex] = pivot
        if pindex < k:
            left = pindex + 1
        else:
            right = pindex
    return (float(a[k - 1]) + float(a[k])) / 2.0 if n % 2 == 0 else float(a[k])


def sd(a, n: int) -> np.float32:
    """siril_stats_float_sd: double sums, float mean, float squared deviations, N-1."""
    acc = 0.0
    for i in range(n):
        acc += float(a[i])
    mean = f32(acc / n)
    acc = 0.0
    for i in range(n):
        d = f32(f32(a[i]) - mean)
        acc += float(f32(d * d))
    return f32(np.sqrt(f32(acc / (n - 1))))


def _clip(pixel, sigma, siglow, sighigh, median) -> int:
    if f32(median - pixel) > f32(sigma * siglow):
        return -1
    if f32(pixel - median) > f32(sigma * sighigh):
        return 1
    return 0


def reject(values: list, mode: str, siglow: float, sighigh: float, crej: list | None = None) -> list:
    """apply_rejection_float on the valid (non-null) values in frame order -> kept values, in Siril's final
    order. crej: optional [low, high] counters, incremented per rejected value like Siril's."""
    stack = [f32(v) for v in values]
    N = len(stack)
    if N <= 1 or mode == "none":
        return stack
    siglow, sighigh = f32(siglow), f32(sighigh)
    crej = crej if crej is not None else [0, 0]
    r = 0
    if mode == "sigma":
        median = quickmedian(stack, N)
        if median == 0.0:
            return []
        firstloop = True
        while True:
            var = sd(stack, N)
            if not firstloop:
                median = quickmedian(stack, N)
            firstloop = False
            out = []
            for i in range(N):
                rej = 0
                if N - r > 4:
                    rej = _clip(stack[i], var, siglow, sighigh, f32(median))
                    if rej:
                        r += 1
                        crej[0 if rej < 0 else 1] += 1
                if not rej:
                    out.append(stack[i])
            changed = len(out) != N
            stack, N = out, len(out)
            if not (changed and N > 3):
                break
        return stack
    while True:                                      # winsorized
        sigma = sd(stack, N)
        medianf = f32(quickmedian(stack, N))
        w = list(stack[:N])
        while True:
            m0 = f32(medianf - f32(f32(1.5) * sigma))
            m1 = f32(medianf + f32(f32(1.5) * sigma))
            w = [min(m1, max(m0, x)) for x in w]
            sigma0 = sigma
            sigma = f32(f32(1.134) * sd(w, N))
            if not (abs(f32(sigma - sigma0)) > f32(sigma0 * f32(0.0005))):
                break
        out = []
        for i in range(N):
            rej = 0
            if N - r > 4:
                rej = _clip(stack[i], sigma, siglow, sighigh, medianf)
                if rej:
                    r += 1
                    crej[0 if rej < 0 else 1] += 1
            if not rej:
                out.append(stack[i])
        changed = len(out) != N
        stack, N = out, len(out)
        if not (changed and N > 3):
            break
    return stack


def pixel(values, weights, mode: str, siglow: float, sighigh: float, weighted: bool,
          crej: list | None = None) -> float:
    """mean_and_reject for one pixel. values/weights: valid samples in frame order
    (weights = drizzle * feather * image weight, already multiplied). crej: as in reject."""
    values = [f32(v) for v in values]
    kept = reject(values, mode, siglow, sighigh, crej)
    if not kept:
        return 0.0
    if not weighted:
        return float(sum(float(v) for v in kept) / len(kept))
    pmin, pmax = min(kept), max(kept)
    s = norm = 0.0
    for v, w in zip(values, weights):
        if pmin <= v <= pmax and v != 0:
            s += float(v) * float(w)
            norm += float(w)
    if norm == 0.0 or s == 0.0:
        s = sum(float(v) for v in values if pmin <= v <= pmax and v > 0)
        return s / len(kept)
    return s / norm
