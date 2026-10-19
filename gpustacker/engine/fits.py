# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Minimal FITS primary-HDU reader/writer (no astropy): headers, raw row access, streamed output."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

BLOCK = 2880

_DTYPES = {-32: ">f4", -64: ">f8", 8: "u1", 16: ">i2", 32: ">i4"}
_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "END"}


def _parse_value(raw: str):
    s = raw.strip()
    if s.startswith("'"):
        m = re.match(r"'((?:[^']|'')*)'", s)
        return (m.group(1).replace("''", "'") if m else s.strip("'")).rstrip()
    v = s.split("/", 1)[0].strip()
    if v in ("T", "F"):
        return v == "T"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v.replace("D", "E"))
    except ValueError:
        return v


@dataclass
class Header:
    """Primary header: raw cards (without END), parsed values and where the data starts."""
    cards: list[str]
    values: dict[str, object]
    data_offset: int
    path: str = ""
    _shape: tuple[int, int, int] | None = field(default=None, repr=False)

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    @property
    def bitpix(self) -> int:
        return int(self.values["BITPIX"])

    @property
    def shape(self) -> tuple[int, int, int]:
        """(channels, height, width)."""
        if self._shape is None:
            naxis = int(self.values.get("NAXIS", 0))
            if naxis not in (2, 3):
                raise ValueError(f"{self.path}: NAXIS={naxis}, expected a 2-D or 3-D image")
            w, h = int(self.values["NAXIS1"]), int(self.values["NAXIS2"])
            c = int(self.values.get("NAXIS3", 1)) if naxis == 3 else 1
            self._shape = (c, h, w)
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        try:
            return np.dtype(_DTYPES[self.bitpix])
        except KeyError:
            raise ValueError(f"{self.path}: unsupported BITPIX {self.bitpix}") from None

    @property
    def data_bytes(self) -> int:
        c, h, w = self.shape
        return c * h * w * self.dtype.itemsize

    def user_cards(self) -> list[str]:
        """Cards other than the structural ones (for copying metadata to a new file)."""
        return [c for c in self.cards if c[:8].strip() not in _STRUCTURAL]


def read_header(path: str) -> Header:
    cards: list[str] = []
    values: dict[str, object] = {}
    with open(path, "rb") as f:
        nblk = 0
        while True:
            blk = f.read(BLOCK)
            if len(blk) < BLOCK:
                raise ValueError(f"{path}: truncated FITS header")
            nblk += 1
            for i in range(0, BLOCK, 80):
                card = blk[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    return Header(cards, values, nblk * BLOCK, path)
                cards.append(card)
                if card[8:10] == "= ":
                    values[key] = _parse_value(card[10:])
            if nblk > 4000:
                raise ValueError(f"{path}: no END card in the header")


class RowReader:
    """Reads top-down row ranges of one plane straight from the file (FITS rows are bottom-up)."""

    def __init__(self, path: str, header: Header | None = None):
        self.header = header or read_header(path)
        self.channels, self.height, self.width = self.header.shape
        self.dtype = self.header.dtype
        self._f = open(path, "rb", buffering=0)

    def raw_rows(self, plane: int, ra: int, rb: int) -> np.ndarray:
        """Undecoded bytes of top-down rows [ra, rb) of `plane`, still in file (bottom-up) order."""
        n, w, h, isz = rb - ra, self.width, self.height, self.dtype.itemsize
        buf = np.empty(n * w * isz, dtype=np.uint8)
        self._f.seek(self.header.data_offset + ((plane * h + (h - rb)) * w) * isz)
        got = self._f.readinto(buf)
        if got != buf.nbytes:
            raise IOError(f"{self._f.name}: short read ({got} of {buf.nbytes} bytes)")
        return buf

    def close(self) -> None:
        self._f.close()


def read_image(path: str) -> tuple[np.ndarray, Header]:
    """Whole image as native float32 (C, H, W), top-down (row 0 = top), BZERO/BSCALE applied."""
    hdr = read_header(path)
    c, h, w = hdr.shape
    raw = np.fromfile(path, dtype=hdr.dtype, count=c * h * w, offset=hdr.data_offset).reshape(c, h, w)
    data = raw.astype(np.float32)
    bscale, bzero = float(hdr.get("BSCALE", 1.0)), float(hdr.get("BZERO", 0.0))
    if bscale != 1.0 or bzero != 0.0:
        data = data * np.float32(bscale) + np.float32(bzero)
    return np.ascontiguousarray(data[:, ::-1, :]), hdr


# ----------------------------------------------------------------------------- writing

def card(key: str, value, comment: str = "") -> str:
    """One 80-character header card."""
    if isinstance(value, bool):
        s = f"{key:<8}= {'T' if value else 'F':>20}"
    elif isinstance(value, str):
        q = "'" + value.replace("'", "''").ljust(8) + "'"
        s = f"{key:<8}= {q:<20}"
    elif isinstance(value, (int, np.integer)):
        s = f"{key:<8}= {int(value):>20d}"
    else:
        v = float(value)
        text = repr(v) if abs(v) < 1e15 else f"{v:.15E}"
        if len(text) > 20:
            text = f"{v:.12E}"
        s = f"{key:<8}= {text:>20}"
    if comment:
        s += f" / {comment}"
    return s[:80].ljust(80)


def history(text: str) -> list[str]:
    """HISTORY cards, wrapping long text."""
    out, width = [], 72
    while text:
        out.append(("HISTORY " + text[:width]).ljust(80))
        text = text[width:]
    return out


def write_fits(path: str, data: np.ndarray, cards: list[str], progress=None) -> None:
    """Write (C, H, W) top-down float32 as a standard bottom-up 32-bit float FITS.

    Streams rows (no full big-endian copy in memory) into `path`.partial, then renames,
    so a cancelled or failed write never leaves a truncated file under the final name."""
    c, h, w = data.shape
    head = [card("SIMPLE", True, "conforms to FITS standard"), card("BITPIX", -32, "32-bit float"),
            card("NAXIS", 3 if c > 1 else 2), card("NAXIS1", w), card("NAXIS2", h)]
    if c > 1:
        head.append(card("NAXIS3", c))
    head += [x.ljust(80)[:80] for x in cards] + ["END".ljust(80)]
    text = "".join(head)
    text = text.ljust(-(-len(text) // BLOCK) * BLOCK)
    tmp = path + ".partial"
    rows_per_chunk = max(1, (32 << 20) // (w * 4))
    total, written = c * h, 0
    try:
        with open(tmp, "wb") as f:
            f.write(text.encode("ascii"))
            for ch in range(c):
                for top in range(h, 0, -rows_per_chunk):        # file order = bottom row first
                    lo = max(0, top - rows_per_chunk)
                    f.write(np.ascontiguousarray(data[ch, lo:top][::-1]).astype(">f4").tobytes())
                    written += top - lo
                    if progress:
                        progress(written, total)
            nbytes = c * h * w * 4
            f.write(b"\0" * (-nbytes % BLOCK))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
