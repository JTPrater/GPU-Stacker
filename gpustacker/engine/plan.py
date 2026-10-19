# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""What a job will have to do, without the GPU: usable frames, cached statistics, frames that still
need measuring, canvas/region and how many bytes each pass reads. Shared by the engine and the UI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from .sequence import Canvas, Frame, Sequence, max_framing
from .stats import FrameStats, StatsCache, file_signature


def output_folder(seq_or_lights_dir: str) -> str:
    """Where results go: the dataset folder (a sequence's folder, or the one above it when it is Siril's
    'process' folder)."""
    d = os.path.normpath(seq_or_lights_dir)
    return os.path.dirname(d) if os.path.basename(d).lower() == "process" else d


def auto_output_path(folder: str, name: str, frames: int, seconds: float, unique: bool = True) -> str:
    """<folder>/<name>_<N>frames_<h>h_gpu_<date_time>.fit, e.g. M_45_906frames_15.1h_gpu_2026-10-13_1739.fit;
    with unique, _2, _3, ... is added when that file already exists."""
    name = "".join(ch for ch in (name or "").replace(" ", "_") if ch.isalnum() or ch in "_-+.") or "stack"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = os.path.join(folder, f"{name}_{frames}frames_{seconds / 3600:.1f}h_gpu_{stamp}")
    path, k = base + ".fit", 2
    while unique and os.path.exists(path):
        path, k = f"{base}_{k}.fit", k + 1
    return path


@dataclass
class StatsInfo:
    stats: dict[int, FrameStats]                    # filenum -> best known stats (Siril's M lines, then cache)
    sigs: dict[int, list]                           # filenum -> file signature
    mask_paths: dict[int, str | None] = field(default_factory=dict)
    cache: StatsCache | None = None


def load_stats(seq: Sequence, frames: list[Frame], need_mask: bool = False) -> StatsInfo:
    """Known statistics for `frames`: Siril's own M-line values first, then this app's cache."""
    layers = seq.layers
    cache = StatsCache(seq.stats_cache_path())
    info = StatsInfo({}, {}, {}, cache)
    for fr in frames:
        path = seq.frame_path(fr)
        try:
            sig = file_signature(path)
        except OSError:
            sig = [0, 0]
        st = cache.get(fr.filenum, sig) or FrameStats()
        sir = fr.siril
        if all(c in sir and sir[c].scale and sir[c].location is not None for c in range(layers)):
            st.loc = [sir[c].location for c in range(layers)]
            st.scale = [sir[c].scale for c in range(layers)]
        if all(c in sir and sir[c].bgnoise for c in range(layers)) and not st.has_noise(layers):
            st.noise = [sir[c].bgnoise for c in range(layers)]
        info.stats[fr.filenum] = st
        info.sigs[fr.filenum] = sig
        if need_mask:
            mp = None
            for cand in (seq.siril_mask_path(fr), seq.own_mask_path(fr)):
                try:
                    if os.stat(cand).st_mtime_ns >= sig[1]:
                        mp = cand
                        break
                except OSError:
                    continue
            info.mask_paths[fr.filenum] = mp
    return info


def needs_measuring(st: FrameStats, layers: int, need_norm: bool, need_noise: bool, need_mask: bool,
                    mask_path: str | None) -> bool:
    if st.bad:
        return False
    return ((need_norm and not st.has_norm(layers)) or (need_noise and not st.has_noise(layers))
            or (need_mask and mask_path is None))


def region_for(canvas: Canvas, crop=None, rows=None) -> tuple[int, int, int, int]:
    x_lo, y_lo, x_hi, y_hi = 0, 0, canvas.width, canvas.height
    if crop:
        cx, cy, cw, ch = (int(v) for v in crop)
        x_lo, y_lo = max(0, cx), max(0, cy)
        x_hi, y_hi = min(canvas.width, cx + cw), min(canvas.height, cy + ch)
    if rows:
        y_lo, y_hi = max(y_lo, int(rows[0])), min(y_hi, int(rows[1]))
    return x_lo, y_lo, x_hi, y_hi


def overlaps(fr: Frame, canvas: Canvas, region) -> bool:
    x_lo, y_lo, x_hi, y_hi = region
    x0, y0 = canvas.x0[fr.filenum], canvas.y0[fr.filenum]
    return x0 < x_hi and x0 + fr.width > x_lo and y0 < y_hi and y0 + fr.height > y_lo


def canvas_origin(frames: list[Frame]) -> tuple[int, int]:
    """Registration coordinates of canvas pixel (0, 0) (Siril: (int)xmin, (int)ymin)."""
    return int(min(fr.tx for fr in frames)), int(min(fr.ty for fr in frames))


def shift_region(region, old_origin, new_origin, canvas: Canvas):
    dx, dy = old_origin[0] - new_origin[0], old_origin[1] - new_origin[1]
    x_lo, y_lo, x_hi, y_hi = region
    return (max(0, x_lo + dx), max(0, y_lo + dy), min(canvas.width, x_hi + dx), min(canvas.height, y_hi + dy))


@dataclass
class Estimate:
    frames: int
    bad: int
    canvas: Canvas | None
    region: tuple
    to_measure: int
    measure_bytes: int
    stack_bytes: int


def estimate(seq: Sequence, frames: list[Frame], info: StatsInfo, normalization: str, weighting: str,
             feather: float, crop=None, rows=None, drizzle: bool = True, bpp: int = 4) -> Estimate:
    layers = seq.layers
    ok = [fr for fr in frames if not info.stats.get(fr.filenum, FrameStats()).bad]
    if len(ok) < 2:
        return Estimate(len(ok), len(frames) - len(ok), None, (0, 0, 0, 0), 0, 0, 0)
    canvas = max_framing(ok)
    region = region_for(canvas, crop, rows)
    need_norm, need_noise, need_mask = normalization == "addscale", weighting == "noise", feather > 0
    ref = seq.reference_frame(ok)
    inside = [fr for fr in ok if overlaps(fr, canvas, region)]
    todo = [fr for fr in inside + ([ref] if ref not in inside else [])
            if needs_measuring(info.stats[fr.filenum], layers, need_norm, need_noise, need_mask,
                               info.mask_paths.get(fr.filenum))]
    planes = layers if (need_norm or need_noise) else 1
    measure = sum(fr.width * fr.height * planes * bpp for fr in todo)
    y_lo, y_hi = region[1], region[3]
    stack = 0
    for fr in inside:
        y0 = canvas.y0[fr.filenum]
        rows_in = max(0, min(y_hi, y0 + fr.height) - max(y_lo, y0))
        stack += rows_in * fr.width * layers * (bpp + (1 if drizzle else 0))
    return Estimate(len(ok), len(frames) - len(ok), canvas, region, len(todo), measure, stack)
