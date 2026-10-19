# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Star-pattern registration (CPU, numpy): triangle matching, robust fit, mosaic chaining.

Each frame gets an affine transform from its top-down continuous pixel coordinates to the
reference frame's. Triangles of bright stars are matched by their side-length ratios (rotation,
shift and scale invariant), the dominant similarity transform wins a vote, and the fit is then
refined on mutual nearest neighbours. Mosaic panels that don't overlap the reference are
registered through already-registered frames taken close in time.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .stars import StarList

TRI_STARS = 45          # brightest stars used to build triangles
TRI_NEIGHBOURS = 6
INV_TOL = 0.006
MIN_INLIERS = 12


@dataclass
class Transform:
    m: np.ndarray       # 2x3 affine: [x', y'] = m[:, :2] @ [x, y] + m[:, 2]

    @staticmethod
    def identity() -> "Transform":
        return Transform(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def apply(self, x, y):
        m = self.m
        return m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]

    def matrix3(self) -> np.ndarray:
        return np.vstack([self.m, [0.0, 0.0, 1.0]])

    def compose(self, other: "Transform") -> "Transform":
        """self ∘ other: apply other, then self."""
        return Transform((self.matrix3() @ other.matrix3())[:2])

    def inverse(self) -> "Transform":
        return Transform(np.linalg.inv(self.matrix3())[:2])

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.m[1, 0], self.m[0, 0])))

    @property
    def scale(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.m[:, :2]))))

    def to_list(self) -> list:
        return [float(v) for v in self.m.ravel()]

    @staticmethod
    def from_list(v) -> "Transform":
        return Transform(np.asarray(v, dtype=np.float64).reshape(2, 3))


@dataclass
class Match:
    t: Transform          # frame -> target
    inliers: int
    rms: float


# ----------------------------------------------------------------------------- triangles

@dataclass
class _Tris:
    verts: np.ndarray     # (T, 3) star indices ordered: opposite longest, middle, shortest side
    inv: np.ndarray       # (T, 2) side ratios
    sign: np.ndarray      # (T,) orientation


def _triangles(x: np.ndarray, y: np.ndarray) -> _Tris:
    n = len(x)
    if n < 3:
        return _Tris(np.zeros((0, 3), int), np.zeros((0, 2)), np.zeros(0))
    p = np.stack([x, y], 1)
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    k = min(TRI_NEIGHBOURS, n - 1)
    nn = np.argsort(d, axis=1)[:, 1:k + 1]
    tris = set()
    for i in range(n):
        for a, b in itertools.combinations(nn[i], 2):
            tris.add(tuple(sorted((i, int(a), int(b)))))
    t = np.array(sorted(tris), dtype=np.int64)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    # side opposite each vertex
    s = np.stack([d[b, c], d[a, c], d[a, b]], 1)
    order = np.argsort(-s, axis=1)
    verts = np.take_along_axis(t, order, axis=1)
    ss = np.take_along_axis(s, order, axis=1)
    keep = (ss[:, 2] > 8.0) & (ss[:, 0] - ss[:, 1] > 0.01 * ss[:, 0]) & (ss[:, 1] - ss[:, 2] > 0.01 * ss[:, 0])
    verts, ss = verts[keep], ss[keep]
    inv = np.stack([ss[:, 1] / ss[:, 0], ss[:, 2] / ss[:, 0]], 1)
    v0, v1, v2 = p[verts[:, 0]], p[verts[:, 1]], p[verts[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    return _Tris(verts, inv, np.sign(cross))


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation, uniform scale, shift) src -> dst: 2x3 matrix (Umeyama)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    var = (a ** 2).sum()
    if var <= 0:
        return np.array([[1.0, 0, 0], [0, 1.0, 0]])
    sxx = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum()
    sxy = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
    scale = np.hypot(sxx, sxy) / var
    th = np.arctan2(sxy, sxx)
    c, s = scale * np.cos(th), scale * np.sin(th)
    r = np.array([[c, -s], [s, c]])
    t = md - r @ ms
    return np.hstack([r, t[:, None]])


def _affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(a, dst, rcond=None)
    return sol.T


def _pairs(pa: np.ndarray, pb: np.ndarray, radius: float):
    """Mutual nearest neighbours within radius: (ia, ib, distances)."""
    if len(pa) == 0 or len(pb) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    d2 = (pa[:, None, 0] - pb[None, :, 0]) ** 2 + (pa[:, None, 1] - pb[None, :, 1]) ** 2
    ib = np.argmin(d2, axis=1)
    ia_back = np.argmin(d2, axis=0)
    ia = np.arange(len(pa))
    ok = (ia_back[ib] == ia) & (d2[ia, ib] <= radius * radius)
    return ia[ok], ib[ok], np.sqrt(d2[ia[ok], ib[ok]])


def refine(a: StarList, b: StarList, t0: Transform, model: str = "affine",
           radii=(5.0, 3.0, 2.0, 1.5)) -> Match | None:
    pa = np.stack([a.x, a.y], 1)
    pb = np.stack([b.x, b.y], 1)
    t = t0
    ia = ib = None
    for r in radii:
        x, y = t.apply(pa[:, 0], pa[:, 1])
        ia, ib, _ = _pairs(np.stack([x, y], 1), pb, r)
        if len(ia) < max(4, MIN_INLIERS // 2):
            return None
        fit = _affine if (model == "affine" and len(ia) >= 12) else _similarity
        t = Transform(fit(pa[ia], pb[ib]))
    x, y = t.apply(pa[ia, 0], pa[ia, 1])
    res = np.hypot(x - pb[ib, 0], y - pb[ib, 1])
    return Match(t, int(len(ia)), float(np.sqrt(np.mean(res ** 2))))


def match(a: StarList, b: StarList, model: str = "affine", max_shift: float | None = None) -> Match | None:
    """Transform taking star list a onto star list b, or None."""
    if len(a) < 6 or len(b) < 6:
        return None
    sa, sb = a.brightest(TRI_STARS), b.brightest(TRI_STARS)
    ta, tb = _triangles(sa.x, sa.y), _triangles(sb.x, sb.y)
    if len(ta.inv) == 0 or len(tb.inv) == 0:
        return None
    d = np.hypot(ta.inv[:, None, 0] - tb.inv[None, :, 0], ta.inv[:, None, 1] - tb.inv[None, :, 1])
    ia, ib = np.nonzero((d < INV_TOL) & (ta.sign[:, None] == tb.sign[None, :]))
    if len(ia) == 0:
        return None
    pa = np.stack([sa.x, sa.y], 1)
    pb = np.stack([sb.x, sb.y], 1)
    src = pa[ta.verts[ia]]                  # (M, 3, 2)
    dst = pb[tb.verts[ib]]
    # batched Umeyama similarity for every candidate triangle pair
    ms, md = src.mean(1), dst.mean(1)
    sa_, db_ = src - ms[:, None], dst - md[:, None]
    var = (sa_ ** 2).sum((1, 2))
    sxx = (sa_[..., 0] * db_[..., 0] + sa_[..., 1] * db_[..., 1]).sum(1)
    sxy = (sa_[..., 0] * db_[..., 1] - sa_[..., 1] * db_[..., 0]).sum(1)
    scale = np.hypot(sxx, sxy) / np.maximum(var, 1e-12)
    rot = np.arctan2(sxy, sxx)
    cs, sn = scale * np.cos(rot), scale * np.sin(rot)
    tx0 = md[:, 0] - (cs * ms[:, 0] - sn * ms[:, 1])
    ty0 = md[:, 1] - (sn * ms[:, 0] + cs * ms[:, 1])
    ok = (scale > 0.9) & (scale < 1.1)
    if not ok.any():
        return None
    p = np.stack([np.degrees(rot), tx0, ty0, np.arange(len(ia))], 1)[ok]
    if max_shift is not None:
        p = p[(np.abs(p[:, 1]) < max_shift) & (np.abs(p[:, 2]) < max_shift)]
        if len(p) == 0:
            return None
    # vote: coarse bins in rotation (2 deg) and the translation of the frame centre (8 px)
    cx, cy = np.median(a.x), np.median(a.y)
    best = None
    th = np.radians(p[:, 0])
    tx = p[:, 1] + np.cos(th) * cx - np.sin(th) * cy      # where the frame centre lands
    ty = p[:, 2] + np.sin(th) * cx + np.cos(th) * cy
    keys = np.stack([np.round(p[:, 0] / 2.0), np.round(tx / 8.0), np.round(ty / 8.0)], 1).astype(np.int64)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    a_c, b_c = a.brightest(250), b.brightest(350)          # cheap check of each candidate
    for cand in order[:4]:                   # try the strongest few vote bins
        members = np.flatnonzero(inverse.ravel() == cand)
        if counts[cand] < 2 and best is not None:
            break
        ks = p[members, 3].astype(int)
        t0 = Transform(_similarity(src[ks].reshape(-1, 2), dst[ks].reshape(-1, 2)))
        m = refine(a_c, b_c, t0, model)
        if m is not None and m.inliers >= MIN_INLIERS and m.rms < 2.0:
            if best is None or m.inliers > best.inliers:
                best = m
            if best.inliers >= 0.3 * min(len(a_c), len(b_c)):
                break
    if best is None:
        return None
    final = refine(a.brightest(600), b, best.t, model, radii=(2.5, 1.5))
    return final if final is not None and final.inliers >= MIN_INLIERS else best


# ----------------------------------------------------------------------------- whole sequence

@dataclass
class RegResult:
    transforms: dict = field(default_factory=dict)    # frame index -> Transform (frame -> reference)
    info: dict = field(default_factory=dict)           # frame index -> (inliers, rms, via)
    failed: dict = field(default_factory=dict)         # frame index -> reason


def pick_reference(metrics: dict) -> int:
    """Best frame: many stars and small FWHM (Siril picks its reference on similar grounds)."""
    best, score = None, -1.0
    for i, m in metrics.items():
        if m.nstars >= 20 and m.fwhm == m.fwhm and m.fwhm > 0:
            s = m.nstars / (m.fwhm ** 2) * (0.5 + 0.5 * m.roundness)
            if s > score:
                best, score = i, s
    if best is None:
        best = max(metrics, key=lambda i: metrics[i].nstars)
    return best


def register_all(stars: dict, ref: int, model: str = "affine", progress=None, cancel=None) -> RegResult:
    """Register every frame (dict index -> StarList) onto frame `ref`."""
    res = RegResult()
    res.transforms[ref] = Transform.identity()
    res.info[ref] = (len(stars[ref]), 0.0, "reference")
    todo = [i for i in sorted(stars) if i != ref]
    total = len(todo)
    done = 0
    pending = []
    for i in todo:                                        # direct to the reference
        if cancel is not None and cancel():
            return res
        m = match(stars[i], stars[ref], model)
        if m is not None:
            res.transforms[i] = m.t
            res.info[i] = (m.inliers, m.rms, "reference")
            done += 1
            if progress:
                progress(done, total)
        else:
            pending.append(i)
    changed = True
    while pending and changed:                            # mosaic: chain through neighbours in time
        changed = False
        still = []
        for i in pending:
            if cancel is not None and cancel():
                return res
            near = sorted(res.transforms, key=lambda j: abs(j - i))[:8]
            ok = False
            for j in near:
                if j == ref:
                    continue
                m = match(stars[i], stars[j], model)
                if m is not None:
                    res.transforms[i] = res.transforms[j].compose(m.t)
                    res.info[i] = (m.inliers, m.rms, f"via frame {j + 1}")
                    ok = changed = True
                    done += 1
                    if progress:
                        progress(done, total)
                    break
            if not ok:
                still.append(i)
        pending = still
    for i in pending:
        res.failed[i] = "couldn't match its stars to the other frames"
    if any(v[2] != "reference" for v in res.info.values()):
        _polish(stars, res, model)
    return res


def _polish(stars: dict, res: RegResult, model: str) -> None:
    """Reduce drift in chained (mosaic) registrations: refit every chained frame against a
    catalogue built from all directly registered frames plus its own neighbours."""
    for _ in range(2):
        cat = _catalog(stars, res, list(res.transforms))
        for i, v in list(res.info.items()):
            if v[2] == "reference":
                continue
            m = refine(stars[i], cat, res.transforms[i], model, radii=(3.0, 2.0, 1.5))
            if m is not None and m.inliers >= MIN_INLIERS:
                res.transforms[i] = m.t
                res.info[i] = (m.inliers, m.rms, v[2])


def _catalog(stars: dict, res: RegResult, frames: list, cell: float = 2.0) -> StarList:
    xs, ys, fl = [], [], []
    for i in frames:
        s = stars[i].brightest(200)
        x, y = res.transforms[i].apply(s.x, s.y)
        xs.append(x)
        ys.append(y)
        fl.append(s.flux)
    x, y, f = np.concatenate(xs), np.concatenate(ys), np.concatenate(fl)
    key = np.stack([np.round(x / cell), np.round(y / cell)], 1).astype(np.int64)
    order = np.argsort(-f)
    _, first = np.unique(key[order], axis=0, return_index=True)
    sel = order[first]
    z = np.zeros(len(sel))
    return StarList(x[sel], y[sel], f[sel], z, z, np.zeros(len(sel), bool))
