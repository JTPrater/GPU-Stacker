# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Auto-stretched PNG of a FITS image (whole, binned, or a 1:1 crop) for quick visual checks.

Usage: .venv\\Scripts\\python tools\\preview_png.py IMAGE.fit OUT.png [--bin N] [--crop X Y W H] [--linked]
"""
import argparse
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpustacker.engine.fits import read_image  # noqa: E402
from gpustacker.ui.stretch import apply_stf, auto_stf  # noqa: E402


def write_png(path: str, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\0" + row.tobytes() for row in rgb)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--bin", type=int, default=1)
    ap.add_argument("--crop", type=int, nargs=4)
    ap.add_argument("--linked", action="store_true")
    a = ap.parse_args()
    data, _ = read_image(a.image)
    params = auto_stf(data, linked=a.linked)
    if a.crop:
        x, y, w, h = a.crop
        data = data[:, y:y + h, x:x + w]
    if a.bin > 1:
        c, h, w = data.shape
        b = a.bin
        data = data[:, :h // b * b, :w // b * b].reshape(c, h // b, b, w // b, b).mean(axis=(2, 4))
    write_png(a.out, apply_stf(data, params))
    print(f"wrote {a.out} {data.shape[2]} x {data.shape[1]}")


if __name__ == "__main__":
    main()
