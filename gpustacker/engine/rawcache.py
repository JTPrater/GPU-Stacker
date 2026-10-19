# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Caches and geometry for raw-light stacking (no GPU, safe for the UI).

<dataset>/gpustacker/
  analysis.json, analysis_stars.npz   per-frame star lists and quality metrics
  registration.json                   reference, per-frame transforms (image -> reference pixels)
  measures.json, masks/*.msk          per-frame normalization statistics and feather masks
Entries are invalidated by the file's size and modification time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

from .lights import Light, dataset_dir
from .stats import CACHE_VERSION as STATS_VERSION

VERSION = 1


def cache_dir(lights_dir: str) -> str:
    return os.path.join(dataset_dir(lights_dir), "gpustacker")


def light_sig(lt: Light) -> list:
    try:
        st = os.stat(lt.path)
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return [0, 0]


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if d.get("version") == VERSION else {}
    except (OSError, ValueError):
        return {}


def _save_json(path: str, d: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, **d}, f, separators=(",", ":"))
    os.replace(tmp, path)


class Analysis:
    """Star lists (x, y, flux, fwhm, roundness, saturated) and metrics per light file name."""

    def __init__(self, cdir: str):
        self.json_path = os.path.join(cdir, "analysis.json")
        self.npz_path = os.path.join(cdir, "analysis_stars.npz")
        d = _load_json(self.json_path)
        self.entries: dict = d.get("frames", {})
        self.stars: dict = {}
        if self.entries:
            try:
                with np.load(self.npz_path, allow_pickle=False) as z:
                    names = list(z["names"])
                    off = z["offsets"]
                    data = z["stars"]
                    for i, n in enumerate(names):
                        self.stars[str(n)] = data[off[i]:off[i + 1]]
            except (OSError, ValueError, KeyError):
                self.entries = {}
                self.stars = {}

    def get(self, lt: Light):
        """(stars array (n, 6), metrics dict) or None."""
        e = self.entries.get(lt.name)
        if not e or e.get("sig") != light_sig(lt) or lt.name not in self.stars:
            return None
        return self.stars[lt.name], e

    def put(self, lt: Light, stars: np.ndarray, metrics: dict) -> None:
        self.entries[lt.name] = {"sig": light_sig(lt), **metrics}
        self.stars[lt.name] = np.asarray(stars, dtype=np.float32).reshape(-1, 6)

    def save(self) -> None:
        names = sorted(n for n in self.entries if n in self.stars)
        arrays = [self.stars[n] for n in names]
        offsets = np.concatenate([[0], np.cumsum([len(a) for a in arrays])]).astype(np.int64)
        data = np.concatenate(arrays) if arrays else np.zeros((0, 6), np.float32)
        os.makedirs(os.path.dirname(self.npz_path), exist_ok=True)
        tmp = self.npz_path + ".tmp.npz"
        np.savez(tmp, names=np.array(names), offsets=offsets, stars=data)
        os.replace(tmp, self.npz_path)
        _save_json(self.json_path, {"frames": {n: self.entries[n] for n in names}})


class Registration:
    def __init__(self, cdir: str):
        self.path = os.path.join(cdir, "registration.json")
        d = _load_json(self.path)
        self.reference: str = d.get("reference", "")
        self.model: str = d.get("model", "")
        self.frames: dict = d.get("frames", {})       # name -> {sig, t, inliers, rms, via}
        self.failed: dict = d.get("failed", {})       # name -> {sig, reason}

    def why_invalid(self, reference: str, model: str, lights: list[Light]) -> str:
        """Why this cache can't be reused for these frames ("" = it can)."""
        if not self.frames:
            return "no saved registration yet"
        if reference != self.reference:
            return f"the reference frame changed ({self.reference or 'none'} -> {reference})"
        if model != self.model:
            return f"the alignment model changed ({self.model or 'none'} -> {model})"
        missing = [lt.name for lt in lights if not (self.frames.get(lt.name) or self.failed.get(lt.name))]
        changed = [lt.name for lt in lights if lt.name not in missing
                   and (self.frames.get(lt.name) or self.failed.get(lt.name)).get("sig") != light_sig(lt)]
        if missing or changed:
            bits = ([f"{len(missing)} new frame(s), e.g. {missing[0]}"] if missing else []) + \
                   ([f"{len(changed)} modified file(s), e.g. {changed[0]}"] if changed else [])
            return ", ".join(bits)
        return ""

    def valid_for(self, reference: str, model: str, lights: list[Light]) -> bool:
        return not self.why_invalid(reference, model, lights)

    def transform(self, name: str):
        e = self.frames.get(name)
        return None if e is None else np.asarray(e["t"], dtype=np.float64).reshape(2, 3)

    def save(self) -> None:
        _save_json(self.path, {"reference": self.reference, "model": self.model, "frames": self.frames,
                               "failed": self.failed})


class Measures:
    def __init__(self, cdir: str):
        self.dir = cdir
        self.path = os.path.join(cdir, "measures.json")
        self.frames: dict = _load_json(self.path).get("frames", {})

    def get(self, lt: Light, key: str) -> dict | None:
        e = self.frames.get(lt.name)
        if not e or e.get("sig") != light_sig(lt) or e.get("key") != key:
            return None
        return e

    def mask_path(self, lt: Light) -> str:
        return os.path.join(self.dir, "masks", os.path.splitext(lt.name)[0] + ".msk")

    def save(self) -> None:
        _save_json(self.path, {"frames": self.frames})


def frame_key(t_box: np.ndarray, w: int, h: int, pixfrac: float) -> str:
    """Identifies a measured warped frame: its transform onto its own box, box size, pixfrac and the version of
    the statistics (new statistics re-measure the frames but keep the analysis and registration)."""
    s = ",".join(f"{v:.6f}" for v in np.asarray(t_box).ravel()) + f"|{w}x{h}|{pixfrac:.4f}|stats{STATS_VERSION}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- geometry

def footprint(t: np.ndarray, width: int, height: int, pixfrac: float = 1.0) -> np.ndarray:
    """(4, 2) corners of a frame's drizzled footprint under 2x3 transform t."""
    pad = 0.5 - pixfrac / 2.0
    xs = np.array([pad, width - pad, width - pad, pad])
    ys = np.array([pad, pad, height - pad, height - pad])
    m = np.asarray(t, dtype=np.float64).reshape(2, 3)
    return np.stack([m[0, 0] * xs + m[0, 1] * ys + m[0, 2], m[1, 0] * xs + m[1, 1] * ys + m[1, 2]], 1)


def raw_canvas(transforms: dict, width: int, height: int, scale: float, pixfrac: float):
    """Canvas for frames with image -> reference transforms (dict key -> 2x3).

    Returns (canvas_w, canvas_h, t_canvas {key: 2x3 image -> canvas}, boxes {key: (x0, y0, w, h)})."""
    if not transforms:
        return 0, 0, {}, {}
    out = {}
    for k, t in transforms.items():
        m = np.asarray(t, dtype=np.float64).reshape(2, 3) * scale
        out[k] = m
    corners = np.concatenate([footprint(m, width, height, pixfrac) for m in out.values()])
    ox, oy = math.floor(corners[:, 0].min() + 1e-9), math.floor(corners[:, 1].min() + 1e-9)
    cw = math.ceil(corners[:, 0].max() - 1e-9) - ox
    ch = math.ceil(corners[:, 1].max() - 1e-9) - oy
    t_canvas, boxes = {}, {}
    for k, m in out.items():
        m = m.copy()
        m[0, 2] -= ox
        m[1, 2] -= oy
        t_canvas[k] = m
        q = footprint(m, width, height, pixfrac)
        x0, y0 = math.floor(q[:, 0].min() + 1e-9), math.floor(q[:, 1].min() + 1e-9)
        x1, y1 = math.ceil(q[:, 0].max() - 1e-9), math.ceil(q[:, 1].max() - 1e-9)
        boxes[k] = (max(0, x0), max(0, y0), min(cw, x1) - max(0, x0), min(ch, y1) - max(0, y0))
    return cw, ch, t_canvas, boxes


def quad_coverage(quads: list, canvas_w: int, canvas_h: int, binning: int) -> np.ndarray:
    """Frames per (binning x binning) cell from footprint quadrilaterals (convex, canvas pixels)."""
    gw, gh = -(-canvas_w // binning), -(-canvas_h // binning)
    diff = np.zeros((gh, gw + 1), np.int32)
    yc = (np.arange(gh) + 0.5) * binning
    for q in quads:
        lo = np.full(gh, np.inf)
        hi = np.full(gh, -np.inf)
        for k in range(4):
            (x1, y1), (x2, y2) = q[k], q[(k + 1) % 4]
            if y1 == y2:
                continue
            m = (yc >= min(y1, y2)) & (yc <= max(y1, y2))
            if not m.any():
                continue
            x = x1 + (yc[m] - y1) / (y2 - y1) * (x2 - x1)
            lo[m] = np.minimum(lo[m], x)
            hi[m] = np.maximum(hi[m], x)
        ok = hi > lo
        if not ok.any():
            continue
        c0 = np.clip(np.ceil(lo[ok] / binning - 0.5), 0, gw).astype(np.int64)
        c1 = np.clip(np.floor(hi[ok] / binning - 0.5) + 1, 0, gw).astype(np.int64)
        rows = np.flatnonzero(ok)
        good = c1 > c0
        np.add.at(diff, (rows[good], c0[good]), 1)
        np.add.at(diff, (rows[good], c1[good]), -1)
    return diff.cumsum(1)[:, :gw]
