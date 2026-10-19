# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
# Contains code ported from Siril; see THIRD-PARTY-NOTICES.txt
"""Siril .seq parsing, per-frame metadata and Siril's -maximize framing.

Pure Python/numpy (no GPU), so the UI can import it.

Coordinates: canvas positions are top-down (row 0 = top), as in Siril's stacking code,
where a registration H gives dx = H[0][2] and dy = -H[1][2] (translation_from_H).
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import os
import re
from dataclasses import dataclass, field

from .fits import read_header

NAN = float("nan")
_NULL = -999999.0
_S_RE = re.compile(r"^S\s+'(.*)'\s+(.*)$")
_EXTENSIONS = (".fit", ".fits", ".fts", ".FIT", ".FITS")


class SequenceError(ValueError):
    """The sequence can't be stacked; the message is meant for the user."""


@dataclass
class LayerStats:
    """Siril's cached statistics for one layer of one image (an M line); None = not computed."""
    ngood: int | None
    median: float | None
    mad: float | None
    location: float | None
    scale: float | None
    bgnoise: float | None


@dataclass
class Frame:
    index: int                      # 0-based position in the .seq (Siril shows index + 1)
    filenum: int
    included: bool
    width: int
    height: int
    tx: float = 0.0                 # registration translation H[0][2]
    ty: float = 0.0                 # registration translation H[1][2]
    pure_translation: bool = True
    has_reg: bool = False
    fwhm: float = NAN
    wfwhm: float = NAN
    roundness: float = NAN
    quality: float = NAN
    background: float = NAN
    nstars: int = 0
    siril: dict = field(default_factory=dict)   # layer -> LayerStats

    @property
    def number(self) -> int:
        return self.index + 1


@dataclass
class FrameMeta:
    """What the frame's FITS header says (exposure, date, size)."""
    exptime: float = 0.0
    livetime: float = 0.0
    stackcnt: int = 1
    date_obs: str = ""
    filter: str = ""
    object: str = ""
    shape: tuple[int, int, int] | None = None
    bitpix: int = 0
    error: str = ""


@dataclass
class Sequence:
    path: str
    name: str
    start: int
    nb_images: int
    nb_selected: int
    fixed_len: int
    reference: int                  # index of the reference image (-1 = not set)
    version: int
    variable_size: bool
    fz: bool
    drizzle: bool
    layers: int
    reg_layer: int | None
    frames: list[Frame]
    ext: str = ".fit"

    @property
    def directory(self) -> str:
        return os.path.dirname(os.path.abspath(self.path))

    def _num(self, fr: Frame) -> str:
        return f"{fr.filenum:0{max(1, self.fixed_len)}d}"

    def frame_path(self, fr: Frame) -> str:
        return os.path.join(self.directory, f"{self.name}{self._num(fr)}{self.ext}")

    def drizzle_path(self, fr: Frame) -> str:
        return os.path.join(self.directory, "drizztmp", f"{self.name}{self._num(fr)}.fit")

    def siril_mask_path(self, fr: Frame) -> str:
        return os.path.join(self.directory, "cache", f"{self.name}{self._num(fr)}.msk")

    def cache_dir(self) -> str:
        """Where this app keeps its own per-sequence cache (stats, generated masks)."""
        return os.path.join(self.directory, "gpustacker")

    def own_mask_path(self, fr: Frame) -> str:
        return os.path.join(self.cache_dir(), "masks", f"{self.name}{self._num(fr)}.msk")

    def stats_cache_path(self) -> str:
        return os.path.join(self.cache_dir(), f"{self.name}stats.json")

    @property
    def included(self) -> list[Frame]:
        return [fr for fr in self.frames if fr.included]

    def reference_frame(self, candidates: list[Frame] | None = None) -> Frame:
        """Siril's reference image if it is among `candidates`, else the first candidate."""
        pool = candidates if candidates is not None else self.frames
        for fr in pool:
            if fr.index == self.reference:
                return fr
        if not pool:
            raise SequenceError("no frames selected")
        return pool[0]


def parse_seq(path: str) -> Sequence:
    """Parse a Siril .seq file (format version 4+ as written by Siril 1.2 / 1.4)."""
    if not os.path.isfile(path):
        raise SequenceError(f"sequence file not found: {path}")
    head = None
    layers = 1
    images: list[tuple[int, bool, int | None, int | None]] = []
    regs: list[tuple[list[float], list[float]]] = []
    stats: dict[tuple[int, int], LayerStats] = {}
    reg_layer = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tag = line.split(None, 1)[0]
            if tag == "S":
                m = _S_RE.match(line)
                if not m:
                    raise SequenceError(f"{os.path.basename(path)}: unreadable S line")
                nums = [int(float(x)) for x in m.group(2).split()]
                nums += [0] * (9 - len(nums))
                head = (m.group(1), nums)
            elif tag == "L":
                layers = int(line.split()[1])
            elif tag == "I":
                p = line.split()
                w = h = None
                if len(p) > 3 and "," in p[3]:
                    w, h = (int(v) for v in p[3].split(","))
                images.append((int(p[1]), p[2] == "1", w, h))
            elif tag[0] == "R" and tag[1:].isdigit():
                layer = int(tag[1:])
                if reg_layer is None:
                    reg_layer = layer
                if layer != reg_layer:
                    continue
                p = line.split()
                if "H" in p:
                    k = p.index("H")
                    regs.append(([float(v) for v in p[1:k]], [float(v) for v in p[k + 1:k + 10]]))
                else:
                    regs.append(([float(v) for v in p[1:]], []))
            elif tag[0] == "M" and "-" in tag:
                layer, idx = (int(v) for v in tag[1:].split("-"))
                # total ngood mean median sigma avgdev mad sqrtbwmv location scale min max norm bgnoise
                p = [float(v) for v in line.split()[1:]]
                if len(p) >= 14:
                    def val(i, positive=False):
                        v = p[i]
                        return None if v == _NULL or (positive and v <= 0) else v
                    stats[(idx, layer)] = LayerStats(
                        ngood=None if p[1] < 0 else int(p[1]), median=val(3), mad=val(6),
                        location=val(8), scale=val(9, True), bgnoise=val(13, True))
    if head is None:
        raise SequenceError(f"{os.path.basename(path)} is not a Siril sequence (no S line)")
    name, (start, nb_images, nb_selected, fixed_len, reference, version, variable, fz, drizzle) = head
    seq = Sequence(path=path, name=name, start=start, nb_images=nb_images, nb_selected=nb_selected,
                   fixed_len=fixed_len, reference=reference, version=version, variable_size=bool(variable),
                   fz=bool(fz), drizzle=bool(drizzle), layers=layers, reg_layer=reg_layer, frames=[])
    for idx, (filenum, incl, w, h) in enumerate(images):
        seq.frames.append(Frame(index=idx, filenum=filenum, included=incl, width=w or 0, height=h or 0))
    if regs and len(regs) != len(images):
        raise SequenceError(f"{len(regs)} registration lines for {len(images)} images in {os.path.basename(path)}")
    for fr, (vals, H) in zip(seq.frames, regs):
        if len(vals) >= 6:
            fr.fwhm, fr.wfwhm, fr.roundness, fr.quality, fr.background = vals[:5]
            fr.nstars = int(vals[5])
        if len(H) == 9:
            fr.has_reg = True
            fr.tx, fr.ty = H[2], H[5]
            fr.pure_translation = (abs(H[0] - 1) < 1e-9 and abs(H[4] - 1) < 1e-9 and abs(H[1]) < 1e-9
                                   and abs(H[3]) < 1e-9 and abs(H[6]) < 1e-12 and abs(H[7]) < 1e-12)
    for (idx, layer), st in stats.items():
        if 0 <= idx < len(seq.frames):
            seq.frames[idx].siril[layer] = st
    seq.ext = _detect_extension(seq)
    if seq.frames and not seq.frames[0].width:
        hdr = read_header(seq.frame_path(seq.frames[0]))
        _, h, w = hdr.shape
        for fr in seq.frames:
            fr.width, fr.height = w, h
    return seq


def _detect_extension(seq: Sequence) -> str:
    if not seq.frames:
        return ".fit"
    fr = seq.frames[0]
    base = os.path.join(seq.directory, f"{seq.name}{fr.filenum:0{max(1, seq.fixed_len)}d}")
    for ext in _EXTENSIONS:
        if os.path.exists(base + ext):
            return ext
    for ext in (".fit.fz", ".fits.fz", ".fz"):
        if os.path.exists(base + ext):
            return ext
    return ".fit"


def check_stackable(seq: Sequence) -> list[str]:
    """Problems that make the sequence impossible to stack here (empty list = fine)."""
    problems = []
    if seq.fz or seq.ext.endswith(".fz"):
        problems.append("The frames are FITS-compressed (.fz). Turn FITS compression off in Siril's "
                        "preferences and re-run the registration.")
    if seq.layers not in (1, 3):
        problems.append(f"{seq.layers} layers per image (only mono or RGB is supported).")
    if not any(fr.has_reg for fr in seq.frames):
        problems.append("The sequence has no registration data. Open the registered sequence (the r_... one "
                        "written by Siril's 'Apply registration'), or open the folder of raw lights instead and "
                        "let GPU Stacker register them.")
    elif any(fr.has_reg and not fr.pure_translation for fr in seq.frames):
        problems.append("The registration isn't applied yet (it contains rotation). Open the r_... "
                        "sequence written by Siril's 'Apply registration' instead.")
    return problems


# ----------------------------------------------------------------------------- headers

def read_frame_meta(path: str) -> FrameMeta:
    m = FrameMeta()
    try:
        hdr = read_header(path)
        m.shape = hdr.shape
        m.bitpix = hdr.bitpix
        m.exptime = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 0.0)) or 0.0)
        live = hdr.get("LIVETIME")
        m.livetime = float(live) if isinstance(live, (int, float)) else m.exptime
        cnt = hdr.get("STACKCNT")
        m.stackcnt = int(cnt) if isinstance(cnt, (int, float)) and cnt > 0 else 1
        m.date_obs = str(hdr.get("DATE-OBS", "") or "")
        m.filter = str(hdr.get("FILTER", "") or "")
        m.object = str(hdr.get("OBJECT", "") or "").strip()
    except (OSError, ValueError) as e:
        m.error = str(e)
    return m


def scan_headers(seq: Sequence, frames: list[Frame] | None = None, workers: int = 8,
                 progress=None) -> dict[int, FrameMeta]:
    """FrameMeta for each frame (keyed by filenum), read in parallel."""
    frames = seq.frames if frames is None else frames
    out: dict[int, FrameMeta] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(read_frame_meta, seq.frame_path(fr)): fr for fr in frames}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            out[futs[fut].filenum] = fut.result()
            if progress and (n % 100 == 0 or n == len(frames)):
                progress(n, len(frames))
    return out


# ----------------------------------------------------------------------------- framing

def round_to_int(x: float) -> int:
    """Siril's round_to_int: half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


@dataclass
class Canvas:
    width: int
    height: int
    x0: dict[int, int]              # filenum -> canvas column of the frame's left edge
    y0: dict[int, int]              # filenum -> canvas row (top-down) of the frame's top row


def max_framing(frames: list[Frame]) -> Canvas:
    """Siril 1.4 `stack -maximize` canvas for these frames (same size and placement)."""
    if not frames:
        raise SequenceError("no frames selected")
    xmin = min(fr.tx for fr in frames)
    ymin = min(fr.ty for fr in frames)
    xmax = max(fr.tx + fr.width for fr in frames)
    ymax = max(fr.ty + fr.height for fr in frames)
    width = int(xmax) - int(xmin) + 1
    height = int(ymax) - int(ymin) + 1
    off_x, off_y = int(xmin), -int(ymin)
    x0, y0 = {}, {}
    for fr in frames:
        x0[fr.filenum] = round_to_int(fr.tx - off_x)
        y0[fr.filenum] = -round_to_int(-fr.ty - off_y)
    return Canvas(width, height, x0, y0)


def coverage_map(frames: list[Frame], canvas: Canvas, binning: int):
    """Number of frames covering each (binning x binning) cell, from frame rectangles."""
    import numpy as np
    gh, gw = -(-canvas.height // binning), -(-canvas.width // binning)
    diff = np.zeros((gh + 1, gw + 1), np.int32)
    for fr in frames:
        xa = max(0, canvas.x0[fr.filenum]) // binning
        ya = max(0, canvas.y0[fr.filenum]) // binning
        xb = min(canvas.width, canvas.x0[fr.filenum] + fr.width) // binning
        yb = min(canvas.height, canvas.y0[fr.filenum] + fr.height) // binning
        if xb <= xa or yb <= ya:
            continue
        diff[ya, xa] += 1
        diff[ya, xb] -= 1
        diff[yb, xa] -= 1
        diff[yb, xb] += 1
    return diff.cumsum(0).cumsum(1)[:gh, :gw]


def largest_rectangle(mask) -> tuple[int, int, int, int]:
    """Largest all-True axis-aligned rectangle in a 2-D bool array: (top, left, height, width)."""
    import numpy as np
    rows, cols = mask.shape
    best = (0, 0, 0, 0)
    heights = np.zeros(cols + 1, np.int64)
    for r in range(rows):
        heights[:-1] = np.where(mask[r], heights[:-1] + 1, 0)
        stack: list[tuple[int, int]] = []
        for c in range(cols + 1):
            start = c
            h = int(heights[c])
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                if sh * (c - s) > best[2] * best[3]:
                    best = (r - sh + 1, s, sh, c - s)
                start = s
            stack.append((start, h))
    return best
