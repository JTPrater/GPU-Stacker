# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Raw light frames: folder scanning, FITS metadata and Bayer (CFA) layout. No GPU needed.

Coordinates are top-down everywhere (row 0 = top of the image as displayed). Like Siril with its
default "ROWORDER from the header, else top-down" setting, camera files without ROWORDER (the
Seestar's) are stored top-down: file row r is image row r. Only ROWORDER = BOTTOM-UP files are
flipped. In image coordinates BAYERPAT always describes the top-left 2x2 block, so the colour of
pixel (x, y) is BAYERPAT[(y % 2) * 2 + x % 2] (plus XBAYROFF / YBAYROFF). Checked against Siril's
own Bayer-drizzled frames of the M31 data: same geometry, identical channel statistics.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import os
from dataclasses import dataclass, field

import numpy as np

from .fits import read_header

FITS_EXT = (".fit", ".fits", ".fts")
CHANNEL = {"R": 0, "G": 1, "B": 2}


@dataclass
class Light:
    index: int                      # 0-based position after sorting by time
    path: str
    width: int = 0
    height: int = 0
    channels: int = 0
    bitpix: int = 0
    bzero: float = 0.0
    bscale: float = 1.0
    bayer: str = ""                 # e.g. "GRBG" for the first stored row; "" = not CFA
    bottom_up: bool = False         # ROWORDER = BOTTOM-UP
    exptime: float = 0.0
    date_obs: str = ""
    filter: str = ""
    object: str = ""
    error: str = ""
    data_offset: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def number(self) -> int:
        return self.index + 1

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def cfa_lut(self) -> np.ndarray | None:
        """(2, 2) int array: channel of image pixel (x, y) is lut[y % 2, x % 2]; None if not CFA."""
        return cfa_lut(self.bayer, self.extra.get("xbayroff", 0), self.extra.get("ybayroff", 0))


def cfa_lut(pattern: str, xoff: int = 0, yoff: int = 0) -> np.ndarray | None:
    p = (pattern or "").strip().upper()
    if len(p) != 4 or any(ch not in CHANNEL for ch in p):
        return None
    lut = np.empty((2, 2), np.int32)
    for y in range(2):
        for x in range(2):
            lut[y, x] = CHANNEL[p[((y + yoff) % 2) * 2 + (x + xoff) % 2]]
    return lut


def find_lights_dir(folder: str) -> str:
    """The folder itself, or its 'lights' subfolder when it has one (the usual Siril layout)."""
    for name in ("lights", "Lights", "LIGHTS", "light", "Light"):
        sub = os.path.join(folder, name)
        if os.path.isdir(sub):
            return sub
    return folder


def dataset_dir(lights_dir: str) -> str:
    """Where results and caches go: the parent of a 'lights' folder, else the folder itself."""
    if os.path.basename(os.path.normpath(lights_dir)).lower() in ("lights", "light"):
        return os.path.dirname(os.path.normpath(lights_dir))
    return lights_dir


def list_fits(folder: str) -> list[str]:
    out = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if f.startswith(".") or not os.path.isfile(p):
            continue
        if os.path.splitext(f)[1].lower() in FITS_EXT:
            out.append(p)
    return sorted(out)


def read_light(path: str, index: int = 0) -> Light:
    lt = Light(index=index, path=path)
    try:
        h = read_header(path)
        c, hh, ww = h.shape
        lt.width, lt.height, lt.channels, lt.bitpix = ww, hh, c, h.bitpix
        lt.bzero, lt.bscale = float(h.get("BZERO", 0.0)), float(h.get("BSCALE", 1.0))
        lt.data_offset = h.data_offset
        if c == 1:
            lt.bayer = str(h.get("BAYERPAT", "") or "").strip().upper()
        lt.bottom_up = str(h.get("ROWORDER", "") or "").strip().upper() == "BOTTOM-UP"
        lt.exptime = float(h.get("EXPTIME", h.get("EXPOSURE", 0.0)) or 0.0)
        lt.date_obs = str(h.get("DATE-OBS", "") or "")
        lt.filter = str(h.get("FILTER", "") or "").strip()
        lt.object = str(h.get("OBJECT", "") or "").strip()
        for k in ("XBAYROFF", "YBAYROFF"):
            v = h.get(k)
            if isinstance(v, (int, float)):
                lt.extra[k.lower()] = int(v)
        for k in ("GAIN", "CCD-TEMP", "IMAGETYP", "EQMODE", "INSTRUME", "CREATOR", "FOCALLEN", "XPIXSZ"):
            if k in h.values:
                lt.extra[k] = h.values[k]
        if lt.bitpix not in (8, 16, -32):
            lt.error = f"BITPIX {lt.bitpix} not supported"
        elif c not in (1, 3):
            lt.error = f"{c} channels not supported"
    except (OSError, ValueError) as e:
        lt.error = str(e)
    return lt


def scan_lights(folder: str, workers: int = 8, progress=None) -> list[Light]:
    """All FITS lights in a folder, sorted by observation time (then name)."""
    files = list_fits(folder)
    lights: list[Light] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(read_light, p) for p in files]
        for n, fut in enumerate(cf.as_completed(futs), 1):
            lights.append(fut.result())
            if progress and (n % 100 == 0 or n == len(files)):
                progress(n, len(files))
    lights.sort(key=lambda lt: (lt.date_obs or "~", lt.path))
    for i, lt in enumerate(lights):
        lt.index = i
    return lights


def common_geometry(lights: list[Light]) -> tuple[int, int, int, str]:
    """The (width, height, channels, bayer) most frames share."""
    from collections import Counter
    good = [lt for lt in lights if not lt.error]
    if not good:
        return 0, 0, 0, ""
    (w, h, c, b), _ = Counter((lt.width, lt.height, lt.channels, lt.bayer) for lt in good).most_common(1)[0]
    return w, h, c, b


# ----------------------------------------------------------------------------- pixel access

def raw_dtype(lt: Light) -> np.dtype:
    return np.dtype({8: "u1", 16: ">i2", -32: ">f4"}[lt.bitpix])


def read_rows_raw(lt: Light, f, plane: int, ra: int, rb: int) -> np.ndarray:
    """Undecoded bytes of image rows [ra, rb) of `plane`, in file order (decode_gpu sorts it out)."""
    isz = raw_dtype(lt).itemsize
    n, w, h = rb - ra, lt.width, lt.height
    first = (h - rb) if lt.bottom_up else ra
    buf = np.empty(n * w * isz, dtype=np.uint8)
    f.seek(lt.data_offset + ((plane * h + first) * w) * isz)
    if f.readinto(buf) != buf.nbytes:
        raise IOError(f"{lt.path}: short read")
    return buf


def read_frame_raw(lt: Light) -> np.ndarray:
    """All undecoded data bytes of the frame (all planes, file order)."""
    isz = raw_dtype(lt).itemsize
    buf = np.empty(lt.width * lt.height * lt.channels * isz, dtype=np.uint8)
    with open(lt.path, "rb", buffering=0) as f:
        f.seek(lt.data_offset)
        if f.readinto(buf) != buf.nbytes:
            raise IOError(f"{lt.path}: short read")
    return buf


def decode_gpu(lt: Light, buf, rows: int, planes: int = 1):
    """Undecoded bytes (file order) -> CuPy float32 (planes, rows, width) in image (top-down) row
    order; integer data scaled to 0..1 (Siril's ushort -> float: / 65535)."""
    import cupy as cp

    from .kernels import bswap_f32, bswap_i16
    w = lt.width
    if lt.bitpix == 16:
        v = bswap_i16(cp.asarray(buf.view(np.uint16)), np.float32(lt.bscale), np.float32(lt.bzero))
    elif lt.bitpix == -32:
        v = bswap_f32(cp.asarray(buf.view(np.uint32)))
    else:
        v = cp.asarray(buf).astype(cp.float32) * np.float32(1.0 / 255.0)
    v = v.reshape(planes, rows, w)
    return v[:, ::-1, :] if lt.bottom_up else v
