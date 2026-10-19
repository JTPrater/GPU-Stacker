# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Pixel comparison of two stacked FITS files (top-left aligned overlap), plus their WCS reference pixels.

Usage: .venv\\Scripts\\python tools\\compare_fits.py A.fit B.fit
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpustacker.engine.fits import read_image  # noqa: E402


def main():
    a, ha = read_image(sys.argv[1])
    b, hb = read_image(sys.argv[2])
    print(f"A {a.shape}  CRPIX {ha.get('CRPIX1')}, {ha.get('CRPIX2')}")
    print(f"B {b.shape}  CRPIX {hb.get('CRPIX1')}, {hb.get('CRPIX2')}")
    c, h, w = (min(x, y) for x, y in zip(a.shape, b.shape))
    a, b = a[:c, :h, :w].astype(np.float64), b[:c, :h, :w].astype(np.float64)
    both = (a != 0) & (b != 0)
    only = (a != 0) ^ (b != 0)
    rel = np.abs(a - b)[both] / np.maximum(np.abs(a[both]), 1e-12)
    print(f"overlap {c} x {h} x {w}: {both.sum()} pixels with data in both, {only.sum()} in only one")
    if rel.size:
        for q in (50, 99, 99.9, 99.99, 100):
            print(f"  relative difference p{q:<6}: {np.percentile(rel, q):.2e}")
        for t in (1e-6, 1e-4, 1e-2):
            print(f"  pixels above {t:.0e}: {(rel > t).sum()}")


if __name__ == "__main__":
    main()
