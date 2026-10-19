# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Compare the engine with numbers Siril itself wrote for a real sequence.

  bgnoise   vs the M lines in the .seq (frames Siril computed statistics for)
  masks     vs Siril's cache/*.msk feather masks

Usage: .venv\\Scripts\\python tools\\check_siril_parity.py path\\to\\r_pp_lights_.seq [frame indices for masks]
"""
import os
import sys
import time

import cupy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpustacker.engine import fits as F  # noqa: E402
from gpustacker.engine import kernels as K  # noqa: E402
from gpustacker.engine.masks import make_mask, read_mask  # noqa: E402
from gpustacker.engine.sequence import parse_seq  # noqa: E402
from gpustacker.engine.stats import measure_planes  # noqa: E402


def load(seq, fr):
    hdr = F.read_header(seq.frame_path(fr))
    c, h, w = hdr.shape
    raw = np.fromfile(seq.frame_path(fr), dtype=np.uint32, count=c * h * w, offset=hdr.data_offset)
    return K.bswap_f32(cp.asarray(raw)).reshape(c, h, w)          # FITS row order


def main():
    seq = parse_seq(sys.argv[1])
    print(f"{len(seq.frames)} frames, {seq.layers} layers, reg layer {seq.reg_layer}, ref {seq.reference}, "
          f"ext {seq.ext}")
    with_noise = [fr for fr in seq.frames if any(s.bgnoise for s in fr.siril.values())]
    for fr in with_noise[:6]:
        planes = load(seq, fr)
        t = time.perf_counter()
        st = measure_planes(planes, need_norm=True, need_noise=True)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t
        for c in range(seq.layers):
            s = fr.siril.get(c)
            ref = s.bgnoise if s else None
            mine = st.noise[c]
            rel = f"{(mine / ref - 1) * 1e6:+9.2f} ppm" if ref and mine else "      n/a"
            print(f"frame #{fr.number:<5} layer {c}: bgnoise {mine if mine is None else f'{mine:.9f}'} vs Siril "
                  f"{ref if ref else 'n/a'!s:>11} {rel} | ngood {st.ngood[c]} vs {s.ngood if s else None}")
        print(f"   (measured in {dt * 1000:.0f} ms)")

    idxs = [int(v) for v in sys.argv[2:]] or [0, 1, len(seq.frames) // 3, len(seq.frames) // 2, len(seq.frames) - 1]
    worst = 0.0
    for idx in idxs:
        fr = seq.frames[idx]
        path = seq.siril_mask_path(fr)
        if not os.path.isfile(path):
            print(f"frame #{fr.number}: no Siril mask")
            continue
        sir = read_mask(path)                                      # top-down
        planes = load(seq, fr)
        t = time.perf_counter()
        mine = make_mask(planes[1 if seq.layers == 3 else 0])[::-1]  # FITS order -> top-down
        dt = time.perf_counter() - t
        if mine.shape != sir.shape:
            print(f"frame #{fr.number}: shape {mine.shape} vs Siril {sir.shape}")
            continue
        diff = np.abs(mine - sir)
        worst = max(worst, float(diff.max()))
        print(f"frame #{fr.number:<5} mask {mine.shape}: max |diff| {diff.max():.6f}, differing pixels "
              f"{int((diff > 0).sum())} of {diff.size}, max distance {sir.max():.3f} ({dt * 1000:.0f} ms)")
    print("worst mask difference:", worst)


if __name__ == "__main__":
    main()
